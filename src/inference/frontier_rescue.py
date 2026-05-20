"""Frontier-oracle rescue.

For each STUDENT prediction whose confidence is below `--conf_threshold`,
re-ask a frontier API (the same prompt the student saw) and replace the
final_prediction.family with the frontier model's answer.

This is selective: only the low-confidence student predictions are sent
to the API, so the cost is bounded.

Inputs:
  --student   student prediction JSONL (with `trace`/`final_prediction`)
  --prompts   prompts JSONL (with `messages`)
  --manifest  manifest restricting which pairs to consider

Output JSONL has the same schema as student prediction; rescued records
get a `frontier_rescue` block.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]


JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def parse_trace(raw: str):
    if not raw:
        return None
    if raw.lstrip().startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n", "", raw.strip())
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    m = JSON_BLOCK_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def call_openai(messages, model, max_tokens, temperature):
    from openai import OpenAI, RateLimitError, APIStatusError
    client = OpenAI()
    backoff = 1.0
    for attempt in range(8):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            return resp.choices[0].message.content
        except RateLimitError as e:
            m = re.search(r"try again in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 0.5 if m else min(backoff, 30.0)
            time.sleep(wait); backoff *= 2.0
        except APIStatusError as e:
            if getattr(e, "status_code", None) == 429:
                time.sleep(min(backoff, 30.0)); backoff *= 2.0
                continue
            raise
    raise RuntimeError("frontier_rescue: rate limit exhausted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--conf_threshold", type=float, default=0.65)
    ap.add_argument("--max_n", type=int, default=200,
                    help="Cap on how many low-conf records to escalate per run.")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--throttle_ms", type=int, default=8000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    keep = set()
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                keep.add(json.loads(line)["pair_id"])

    prompts = {}
    with open(args.prompts) as f:
        for line in f:
            r = json.loads(line)
            pid = r["pair_id"]
            if pid in keep:
                prompts[pid] = r["messages"]

    out_path = Path(args.output)
    done = set()
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["pair_id"])
                except Exception:
                    pass

    eligible = []
    student_records = {}
    with open(args.student) as f:
        for line in f:
            r = json.loads(line)
            pid = r["pair_id"]
            if (r.get("input_order") or "ab") != "ab":
                continue
            if pid in student_records:
                continue
            student_records[pid] = r
            if pid in done:
                continue
            if pid not in prompts:
                continue
            fp = r.get("final_prediction") or {}
            try:
                conf = float(fp.get("confidence", 1.0) or 1.0)
            except Exception:
                conf = 1.0
            if conf < args.conf_threshold or fp.get("abstain"):
                eligible.append((pid, conf))

    eligible.sort(key=lambda x: x[1])  # lowest conf first
    eligible = eligible[: args.max_n]
    print(f"[frescue] {len(eligible)} student records to escalate (conf<{args.conf_threshold})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_changed = 0
    with out_path.open("a") as fout:
        # Write any record not eligible — but only write ALL student records on
        # first pass.  We always emit every pair_id, rescued or not.
        if not args.resume or out_path.stat().st_size == 0:
            for pid, r in student_records.items():
                if pid not in {p for p, _ in eligible}:
                    fout.write(json.dumps(r) + "\n")
                    done.add(pid)

        for i, (pid, conf) in enumerate(eligible, 1):
            messages = prompts[pid]
            try:
                raw = call_openai(messages, args.model, args.max_tokens, args.temperature)
            except Exception as e:
                print(f"[frescue] {pid} failed: {e}", file=sys.stderr)
                continue
            trace = parse_trace(raw)
            new_fam = None
            if trace and "final_answer" in trace:
                fa = trace["final_answer"]
                if fa.get("family") in FAMS:
                    new_fam = fa["family"]
            r = dict(student_records[pid])
            if new_fam:
                old = (r.get("final_prediction") or {}).get("family")
                r.setdefault("frontier_rescue", {})
                r["frontier_rescue"]["model"] = args.model
                r["frontier_rescue"]["from"] = old
                r["frontier_rescue"]["to"] = new_fam
                r["final_prediction"]["family"] = new_fam
                if trace["final_answer"].get("subtype"):
                    r["final_prediction"]["subtype"] = trace["final_answer"]["subtype"]
                n_changed += 1
            fout.write(json.dumps(r) + "\n")
            fout.flush()
            if i % 10 == 0:
                print(f"[frescue] {i}/{len(eligible)} changed={n_changed}")
            time.sleep(args.throttle_ms / 1000.0)

    print(f"[frescue] done. {n_changed}/{len(eligible)} changed; wrote {out_path}")


if __name__ == "__main__":
    main()
