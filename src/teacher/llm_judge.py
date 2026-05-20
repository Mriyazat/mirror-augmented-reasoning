"""B2.5 — Out-of-family LLM-as-judge (4th judge layer).

Earlier work hit a wall with 3 LLM-judges of the *same family* (all trained on
similar web text, all sharing biases with the teacher).  Our work adds ONE
judge from a DIFFERENT model family (OpenAI GPT-4o or Anthropic Claude)
run on a ~2k held-out sample.  It is a diagnostic, not a gate:

    - INPUT:  a teacher trace that *passed* QC
    - OUTPUT: {"verdict": "agree"|"disagree"|"abstain", "reason": "..."}
    - USE:    calibration signal for the paper — agreement rate between
              (rule-based QC ∩ PRM) and an out-of-family LLM.  A high
              agreement rate validates our pipeline; a low one flags
              shared bias between the 3 open-source teachers.

It NEVER removes a trace from `teacher_clean.jsonl`.  The critic (B4)
still makes keep/drop decisions.

Usage:

    python -m src.teacher.llm_judge \\
        --qc outputs/teacher/qc_subset25k_openai-*.jsonl \\
        --sample_size 2000 --provider openai --model gpt-4o

Writes:
    outputs/teacher/judge_<split>_<model>.jsonl          per-record verdicts
    outputs/teacher/judge_<split>_<model>.summary.md     agreement table
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("DDI_OUTPUTS", ROOT / "outputs")) / "teacher"


JUDGE_SYSTEM_PROMPT = """You are an expert pharmacologist acting as an independent
reviewer of a drug-drug-interaction (DDI) reasoning trace produced by another LLM.
Your ONLY job: decide whether the trace's `final_answer` is consistent with the
trace's own reasoning steps AND with the provided QUERY PAIR.  You have no access
to a gold label, so judge the *internal* quality of the trace.

Return a JSON object with exactly these keys:
  - verdict: "agree" | "disagree" | "abstain"
  - reason:  1-2 sentences explaining your verdict

Guidance:
  - "agree"    — final_answer is consistent with reasoning, direction is
                 sensible, summary names both drugs, no red flags.
  - "disagree" — final_answer contradicts reasoning, direction flipped,
                 summary nonsensical, or trace hallucinates claims not
                 supported by its own steps.
  - "abstain"  — trace is too vague/ambiguous to judge confidently.
"""


def build_judge_user_prompt(rec: dict) -> str:
    pid = rec["pair_id"]
    parsed = rec.get("parsed") or {}
    fa = parsed.get("final_answer", {})
    steps = parsed.get("steps", [])
    step_lines = []
    for s in steps:
        step_lines.append(
            f"  Step {s.get('step_id')}. [{s.get('role')}, {s.get('direction_tag')}] "
            f"{s.get('claim')}  "
            f"(evidence: {', '.join(s.get('evidence_ids', []) or [])})"
        )
    return f"""QUERY PAIR
  pair_id = {pid}

REASONING STEPS
{chr(10).join(step_lines)}

FINAL ANSWER
  family     = {fa.get('family')}
  subtype    = {fa.get('subtype')}
  direction  = {fa.get('direction_tag')}
  polarity   = {fa.get('polarity')}
  abstain    = {fa.get('abstain')}
  confidence = {fa.get('confidence')}
  summary    = {fa.get('summary')}

TASK
  Return {{"verdict": "agree"|"disagree"|"abstain", "reason": "..."}}
"""


def _judge_openai(model: str, api_key: str | None, base_url: str | None,
                  system: str, user: str, timeout_s: int = 60) -> dict:
    import requests
    headers = {
        "Authorization": f"Bearer {api_key or os.environ.get('OPENAI_API_KEY', '')}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }
    url = (base_url or "https://api.openai.com/v1") + "/chat/completions"
    r = requests.post(url, headers=headers, json=body, timeout=timeout_s)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except Exception:
        return {"verdict": "abstain", "reason": f"judge returned unparseable JSON: {content[:200]}"}


def _judge_anthropic(model: str, api_key: str | None,
                     system: str, user: str, timeout_s: int = 60) -> dict:
    import requests
    headers = {
        "x-api-key": api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers=headers, json=body, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    content = data["content"][0]["text"] if data.get("content") else ""
    try:
        # Find the first JSON object
        import re
        m = re.search(r"\{.*\}", content, re.DOTALL)
        return json.loads(m.group(0)) if m else {
            "verdict": "abstain", "reason": f"no JSON in response: {content[:200]}"
        }
    except Exception:
        return {"verdict": "abstain", "reason": f"unparseable: {content[:200]}"}


def run_judge(qc_path: Path, provider: str, model: str,
              sample_size: int, api_key: str | None, base_url: str | None,
              seed: int = 42) -> Path:
    qc_path = qc_path.resolve()
    print(f"[judge] reading {qc_path.relative_to(ROOT) if ROOT in qc_path.parents else qc_path}")

    # Only judge records that *passed* critical QC; we're calibrating
    # agreement on the traces we think are good.
    kept: list[dict] = []
    with qc_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("critical_passed") or r.get("passed"):
                kept.append(r)
    print(f"[judge] {len(kept):,} QC-passing candidates available")

    rng = random.Random(seed)
    if len(kept) > sample_size:
        rng.shuffle(kept)
        kept = kept[:sample_size]
    print(f"[judge] sampling {len(kept):,} for judging (seed={seed})")

    out_path = OUT_DIR / f"judge_{qc_path.stem}_{provider}-{model.replace('/', '_')}.jsonl"
    summary_path = out_path.with_suffix(".summary.md")

    verdicts = Counter()
    t0 = time.time()
    n_err = 0
    with out_path.open("w") as fout:
        for i, rec in enumerate(kept):
            user_p = build_judge_user_prompt(rec)
            try:
                if provider == "openai":
                    verdict = _judge_openai(model, api_key, base_url,
                                            JUDGE_SYSTEM_PROMPT, user_p)
                elif provider == "anthropic":
                    verdict = _judge_anthropic(model, api_key,
                                               JUDGE_SYSTEM_PROMPT, user_p)
                else:
                    raise ValueError(f"unknown judge provider '{provider}'")
            except Exception as e:
                verdict = {"verdict": "abstain", "reason": f"provider error: {e}"}
                n_err += 1
            v = verdict.get("verdict", "abstain").lower()
            verdicts[v] += 1
            fout.write(json.dumps({
                "pair_id": rec["pair_id"],
                "candidate_id": rec["candidate_id"],
                "teacher_id": rec.get("teacher_id"),
                "judge_model": f"{provider}:{model}",
                "verdict": v,
                "reason": verdict.get("reason", ""),
            }) + "\n")
            if (i + 1) % 100 == 0:
                rate = (i + 1) / max(1, time.time() - t0)
                eta = (len(kept) - i - 1) / max(1e-9, rate)
                print(f"[judge]   {i+1:,}/{len(kept):,}  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)

    total = sum(verdicts.values())
    lines = [
        f"# LLM-judge summary — {qc_path.name}",
        "",
        f"- judge: {provider}:{model}",
        f"- sample: {total:,} QC-passing candidates",
        f"- provider errors: {n_err}",
        "",
        "| Verdict | Count | Rate |",
        "|---------|------:|-----:|",
    ]
    for v in ["agree", "disagree", "abstain"]:
        c = verdicts.get(v, 0)
        lines.append(f"| {v} | {c:,} | {100*c/max(1,total):.1f}% |")
    lines.append("")
    lines.append("**Interpretation:** `agree` rate is our key diagnostic.  ≥ 85% → "
                 "rule-based QC + PRM are consistent with an out-of-family expert.  "
                 "< 70% → shared bias between the open-source teachers; revisit "
                 "the prompt / retrieval / PRM.")
    summary_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[judge] per-record verdicts: {out_path.relative_to(ROOT) if ROOT in out_path.parents else out_path}")
    print(f"[judge] summary:              {summary_path.relative_to(ROOT) if ROOT in summary_path.parents else summary_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--qc", required=True, help="Path to qc_<split>_<provider>.jsonl")
    p.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    p.add_argument("--model", default="gpt-4o",
                   help="e.g. gpt-4o, gpt-4o-mini, claude-3-5-sonnet-20241022")
    p.add_argument("--sample_size", type=int, default=2000)
    p.add_argument("--api_key", default=None)
    p.add_argument("--base_url", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_judge(Path(args.qc), args.provider, args.model,
              args.sample_size, args.api_key, args.base_url, args.seed)
