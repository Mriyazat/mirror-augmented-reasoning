"""Cross-judge LLM-as-judge evaluator for DDI reasoning trace quality.

Sets up a blind, cross-judged comparison:

  Trace from        Judged by
  ---------------   --------------------
  Ours (student)    Claude + GPT-4o + Gemini  (mean of 3)
  Claude            GPT-4o + Gemini           (mean of 2)
  GPT-4o            Claude + Gemini           (mean of 2)
  Gemini            Claude + GPT-4o           (mean of 2)

Each judge scores a trace on 6 rubric dimensions (see trace_quality_rubric.py).
Identity is stripped before prompting (judge cannot tell which model produced
the trace). Output: a long-form CSV (one row per trace x judge x dim) plus a
checkpointed JSONL of full judgments for auditing.

Usage:
    source .env.trace_eval
    python -m src.evaluation.run_trace_quality_xjudge \\
        --pair_ids outputs/diag2/trace_quality/sampled_pair_ids.json \\
        --traces ours=outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_ab.jsonl \\
        --traces claude=outputs/diag2/trace_quality/pred_claude_traces_200.jsonl \\
        --traces gpt4o=outputs/diag2/trace_quality/pred_gpt4o_traces_200.jsonl \\
        --traces gemini=outputs/diag2/trace_quality/pred_gemini_traces_200.jsonl \\
        --prompts outputs/eval_prompts/random_full_test.prompts.jsonl \\
        --out_dir outputs/diag2/trace_quality \\
        --resume
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import inference helpers (call_anthropic, call_openai, call_google).
from src.inference.predict_with_frontier_llm_full_traces import (
    call_anthropic,
    call_google,
    call_openai,
)
from src.evaluation.trace_quality_rubric import (
    RUBRIC_DIMENSIONS,
    build_judge_system_prompt,
    build_judge_user_prompt,
)

DIM_IDS = [d["id"] for d in RUBRIC_DIMENSIONS]


# --------------------------------------------------------------------------
# Cross-judge map: which judges score which models
# --------------------------------------------------------------------------
JUDGE_CONFIG = {
    "claude":  {"provider": "anthropic", "model": "claude-sonnet-4-5",  "in_$/M": 3.00, "out_$/M": 15.00},
    "gpt4o":   {"provider": "openai",    "model": "gpt-4o",             "in_$/M": 2.50, "out_$/M": 10.00},
    "gemini":  {"provider": "google",    "model": "gemini-2.5-flash",   "in_$/M": 0.30, "out_$/M": 2.50},
}

CROSS_JUDGE_MAP = {
    "ours":    ["claude", "gpt4o", "gemini"],
    "claude":  ["gpt4o", "gemini"],
    "gpt4o":   ["claude", "gemini"],
    "gemini":  ["claude", "gpt4o"],
}


# --------------------------------------------------------------------------
# Trace anonymization & rendering
# --------------------------------------------------------------------------
def render_trace_for_judge(trace_obj) -> str:
    """Pretty-print a trace JSON in a clean format the judge can read.

    Strips any model-identifying keywords if present (none expected in our
    schema, but defensive).
    """
    if trace_obj is None:
        return "[NO TRACE OR PARSE FAILED]"
    if isinstance(trace_obj, str):
        try:
            trace_obj = json.loads(trace_obj)
        except Exception:
            return trace_obj[:6000]
    out_lines = []
    steps = trace_obj.get("steps", [])
    out_lines.append(f"REASONING STEPS ({len(steps)} steps total):")
    for s in steps:
        sid = s.get("step_id", "?")
        role = s.get("role", "?")
        claim = s.get("claim", "")
        ev = s.get("evidence_ids", []) or []
        dt = s.get("direction_tag", "n/a")
        fh = s.get("family_hint", "n/a")
        out_lines.append(f"  Step {sid} [role={role}, dir={dt}, fam_hint={fh}]")
        out_lines.append(f"    claim: {claim}")
        out_lines.append(f"    evidence_ids: {ev}")

    fa = trace_obj.get("final_answer", {}) or {}
    out_lines.append("")
    out_lines.append("FINAL ANSWER:")
    out_lines.append(f"  family: {fa.get('family')}")
    out_lines.append(f"  subtype: {fa.get('subtype')}")
    out_lines.append(f"  direction_tag: {fa.get('direction_tag')}")
    out_lines.append(f"  polarity: {fa.get('polarity')}")
    out_lines.append(f"  abstain: {fa.get('abstain')}")
    out_lines.append(f"  confidence: {fa.get('confidence')}")
    summary = fa.get("summary")
    if summary:
        out_lines.append(f"  summary: {summary}")
    return "\n".join(out_lines)


def extract_trace_for_pair(record: dict) -> dict | None:
    """Extract the trace JSON object from a prediction record (handles all
    schema variants we've seen: 'trace' dict, 'trace' str, 'raw_output' str)."""
    if record.get("trace") and isinstance(record["trace"], dict):
        return record["trace"]
    if record.get("trace") and isinstance(record["trace"], str):
        try:
            t = record["trace"]
            if isinstance(t, str) and t.startswith("'"):
                # python repr
                t = eval(t)
            if isinstance(t, dict):
                return t
            if isinstance(t, str):
                return json.loads(t)
        except Exception:
            pass
    raw = record.get("raw_output")
    if raw:
        try:
            from src.inference.predict_with_frontier_llm_full_traces import parse_trace
            return parse_trace(raw)
        except Exception:
            return None
    return None


def render_query_snippet(prompts_dict: dict, pair_id: str, max_chars: int = 4000) -> str:
    """Pull the user message of the original DDI query, truncate to fit."""
    msgs = prompts_dict.get(pair_id, [])
    user_msg = next((m["content"] for m in msgs if m["role"] != "system"), "")
    if len(user_msg) <= max_chars:
        return user_msg
    return user_msg[:max_chars] + "\n...[query truncated]"


# --------------------------------------------------------------------------
# Judge JSON parsing (robust to fenced blocks, trailing commas, etc.)
# --------------------------------------------------------------------------
JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def parse_judge_response(raw: str) -> dict | None:
    if not raw:
        return None
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n", "", txt)
        if txt.rstrip().endswith("```"):
            txt = txt.rstrip()[:-3]
    m = JSON_BLOCK_RE.search(txt)
    if not m:
        return None
    blob = m.group(0)
    for attempt in range(3):
        try:
            obj = json.loads(blob)
            if "scores" not in obj:
                return None
            return obj
        except Exception:
            blob = re.sub(r",\s*}", "}", blob)
            blob = re.sub(r",\s*]", "]", blob)
            blob = re.sub(r"//[^\n]*\n", "\n", blob)
    return None


def validate_scores(obj: dict) -> dict | None:
    sc = obj.get("scores", {})
    out = {}
    for d in DIM_IDS:
        v = sc.get(d)
        if v is None:
            return None
        try:
            v = int(round(float(v)))
        except Exception:
            return None
        if v < 0 or v > 8:
            return None
        out[d] = v
    return out


# --------------------------------------------------------------------------
# Main runner
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_ids", required=True, help="JSON list of pair_ids.")
    ap.add_argument(
        "--traces", action="append", required=True,
        help="Format: MODEL=PATH. Repeat for each model.",
    )
    ap.add_argument("--prompts", required=True, help="Original DDI query prompts JSONL.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models_only", nargs="*", default=None,
                    help="If set, only judge these MODEL keys (e.g. ours claude).")
    ap.add_argument("--judges_only", nargs="*", default=None,
                    help="If set, only use these JUDGE keys (e.g. claude gpt4o).")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="Number of concurrent API calls (across all providers).")
    args = ap.parse_args()

    with open(args.pair_ids) as f:
        pair_ids = json.load(f)
    print(f"[xjudge] {len(pair_ids)} pair_ids loaded")

    # Load traces by model
    traces_by_model = {}
    for spec in args.traces:
        if "=" not in spec:
            raise SystemExit(f"Bad --traces format: {spec} (expected MODEL=PATH)")
        model_key, fp = spec.split("=", 1)
        recs = {}
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                pid = r.get("pair_id")
                if pid in pair_ids:
                    recs[pid] = r
        traces_by_model[model_key] = recs
        print(f"[xjudge] loaded {len(recs)} traces for model='{model_key}' from {fp}")

    # Load prompts
    prompts_dict = {}
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("pair_id")
            if pid in pair_ids:
                prompts_dict[pid] = r.get("messages", [])
    print(f"[xjudge] loaded {len(prompts_dict)} query prompts")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = out_dir / "xjudge_judgments.jsonl"
    csv_path = out_dir / "xjudge_scores_long.csv"

    # Resume: load already-done (model, pair_id, judge) triples
    done = set()
    if args.resume and judgments_path.exists():
        with judgments_path.open() as f:
            for line in f:
                try:
                    j = json.loads(line)
                    done.add((j["model"], j["pair_id"], j["judge"]))
                except Exception:
                    pass
    print(f"[xjudge] resume: {len(done)} judgments already complete")

    system_prompt = build_judge_system_prompt()

    # Build the work list
    work = []
    for model_key, judges in CROSS_JUDGE_MAP.items():
        if args.models_only and model_key not in args.models_only:
            continue
        recs = traces_by_model.get(model_key, {})
        for pid in pair_ids:
            if pid not in recs:
                continue
            for judge_key in judges:
                if args.judges_only and judge_key not in args.judges_only:
                    continue
                key = (model_key, pid, judge_key)
                if key in done:
                    continue
                work.append(key)

    rng = random.Random(args.seed)
    rng.shuffle(work)
    print(f"[xjudge] {len(work)} judgments to do "
          f"(cross-judge over {len(CROSS_JUDGE_MAP)} models x ~2-3 judges each)")

    # Quick cost preview
    avg_in_tok = 4500  # system + query + trace text -> empirical estimate
    avg_out_tok = 700
    by_judge = {}
    for (_m, _p, j) in work:
        by_judge[j] = by_judge.get(j, 0) + 1
    cost_est = 0.0
    for jk, n in by_judge.items():
        cfg = JUDGE_CONFIG[jk]
        cost = (avg_in_tok * cfg["in_$/M"] + avg_out_tok * cfg["out_$/M"]) / 1e6 * n
        cost_est += cost
        print(f"[xjudge] est. {jk}: {n} calls -> ~${cost:.2f}")
    print(f"[xjudge] TOTAL estimated cost: ~${cost_est:.2f}")

    # ------------------------------------------------------------------
    # Concurrent execution with ThreadPoolExecutor (API calls are I/O bound)
    # ------------------------------------------------------------------
    write_lock = threading.Lock()
    counter = {"ok": 0, "fail": 0, "done": 0}
    t0 = time.time()
    fjudg = judgments_path.open("a")

    def _do_one(triple):
        model_key, pid, judge_key = triple
        rec = traces_by_model[model_key].get(pid)
        if rec is None:
            return None
        trace_obj = extract_trace_for_pair(rec)
        trace_text = render_trace_for_judge(trace_obj)
        query_snip = render_query_snippet(prompts_dict, pid)
        user_prompt = build_judge_user_prompt(query_snip, trace_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        cfg = JUDGE_CONFIG[judge_key]
        caller = {"anthropic": call_anthropic, "openai": call_openai, "google": call_google}[cfg["provider"]]
        try:
            if cfg["provider"] == "google":
                # Disable thinking for judge calls (it eats output budget and
                # truncates the JSON response). Judge output is short JSON, so
                # we don't need thinking.
                raw, usage = caller(messages, cfg["model"], args.max_tokens, args.temperature, thinking_budget=0)
            else:
                raw, usage = caller(messages, cfg["model"], args.max_tokens, args.temperature)
        except Exception as e:
            judgment = {
                "model": model_key, "pair_id": pid, "judge": judge_key,
                "error": str(e)[:500],
            }
            with write_lock:
                fjudg.write(json.dumps(judgment) + "\n")
                fjudg.flush()
                counter["fail"] += 1
                counter["done"] += 1
            return False
        parsed = parse_judge_response(raw)
        scores = validate_scores(parsed) if parsed else None
        judgment = {
            "model": model_key, "pair_id": pid, "judge": judge_key,
            "scores": scores,
            "justifications": (parsed or {}).get("justifications") if parsed else None,
            "length_bias_self_check": (parsed or {}).get("length_bias_self_check") if parsed else None,
            "raw_response": raw[:4000] if scores is None else None,
            "usage": usage,
        }
        with write_lock:
            fjudg.write(json.dumps(judgment) + "\n")
            fjudg.flush()
            if scores is None:
                counter["fail"] += 1
            else:
                counter["ok"] += 1
            counter["done"] += 1
            i = counter["done"]
            if i % 25 == 0 or i == len(work):
                rate = i / max(time.time() - t0, 1.0)
                eta_min = (len(work) - i) / max(rate, 0.01) / 60
                print(f"[xjudge] {i}/{len(work)} ok={counter['ok']} fail={counter['fail']} "
                      f"({rate:.2f}/s, ETA {eta_min:.1f}min)", flush=True)
        return True

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_do_one, t) for t in work]
            for _ in as_completed(futures):
                pass
    finally:
        fjudg.close()

    print(f"[xjudge] done. ok={counter['ok']} fail={counter['fail']}; "
          f"elapsed={(time.time()-t0)/60:.1f}min")

    # Write the long-form CSV
    write_long_csv(judgments_path, csv_path)
    print(f"[xjudge] wrote long-form CSV: {csv_path}")


def write_long_csv(judgments_path: Path, csv_path: Path) -> None:
    """Re-read all judgments and write a long-form CSV:
    columns = [model, pair_id, judge, dim, score, justification].
    """
    with judgments_path.open() as f, csv_path.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["model", "pair_id", "judge", "dim", "score", "justification"])
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            scores = j.get("scores")
            if not scores:
                continue
            justs = j.get("justifications") or {}
            for dim in DIM_IDS:
                if dim in scores:
                    writer.writerow([
                        j["model"], j["pair_id"], j["judge"], dim,
                        scores[dim], (justs.get(dim) or "")[:500]
                    ])


if __name__ == "__main__":
    main()
