"""Build a TRACE-ALIGNMENT SFT dataset from student predictions.

Failure mode addressed:
  The student's reasoning trace correctly identifies a family (the
  majority `family_hint` over its trace steps agrees with the gold
  family), but the final_prediction.family disagrees with that majority
  and is wrong.  ~500 such pairs per split (see
  outputs/diag2/qual/reasoning_vs_final_*.json).

The builder takes one or more prediction JSONLs (with full `trace` and
`final_prediction`) plus the matching prompt JSONL (system+user
messages) and writes SFT records in the format the current SFT trainer
already consumes:

  {
    "pair_id":       str,
    "family":        str,     # gold family
    "direction_tag": str,     # gold direction
    "sample_weight": float,   # weighted up for harder rescues
    "tier":          "trace_align",
    "context_ids":   [...],
    "messages":      [
      {"role":"system","content":...},
      {"role":"user","content":...},
      {"role":"assistant","content": <patched JSON output>}
    ]
  }

The patched assistant message keeps the original trace verbatim but
overrides `final_prediction.family` (and subtype / direction / polarity
when gold is available) to be coherent with the trace majority + gold
label.  Optional: filter so we only include rescues whose trace
majority already agrees with gold (otherwise we'd be teaching the
model to lie).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]


def _load_labels():
    rows = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet").to_pylist()
    return {r["pair_id"]: r for r in rows}


def _load_prompts(path: str):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["pair_id"]] = r
    return out


def _trace_majority(trace):
    hints = []
    if not isinstance(trace, dict):
        return None
    for s in trace.get("steps") or []:
        h = s.get("family_hint")
        if h in FAMS:
            hints.append(h)
    if not hints:
        return None
    return Counter(hints).most_common(1)[0][0]


def build_record(prompt_rec, pred_rec, label):
    """Construct an SFT record with the final answer overridden to gold."""
    raw = pred_rec.get("raw_output") or ""
    fp = pred_rec.get("final_prediction") or {}
    new_fp = {
        "family": label["family"],
        "subtype": label.get("subtype") or fp.get("subtype"),
        "direction_tag": label.get("direction_tag") or fp.get("direction_tag"),
        "polarity": label.get("polarity") or fp.get("polarity"),
        "abstain": False,
        "confidence": min(max(float(fp.get("confidence", 0.85)), 0.85), 0.97),
        "label_dist": fp.get("label_dist", {}),
    }

    asst = None
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "final_prediction" in obj:
                obj["final_prediction"] = new_fp
                asst = json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass

    if asst is None:
        obj = {
            "trace": pred_rec.get("trace") or {"steps": []},
            "final_prediction": new_fp,
        }
        asst = json.dumps(obj, ensure_ascii=False)

    messages = list(prompt_rec["messages"])
    messages.append({"role": "assistant", "content": asst})
    return {
        "pair_id": pred_rec["pair_id"],
        "family": label["family"],
        "direction_tag": label.get("direction_tag", "n/a"),
        "sample_weight": 1.0,
        "tier": "trace_align",
        "context_ids": prompt_rec.get("context_ids", []),
        "messages": messages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_jsonl", action="append", required=True,
                    help="path to the with_neighbors prompts file (can repeat)")
    ap.add_argument("--pred_jsonl",   action="append", required=True,
                    help="path to the student prediction JSONL (can repeat, paired by index)")
    ap.add_argument("--output",       required=True)
    ap.add_argument("--mode", choices=["trace_correct", "any_wrong"], default="trace_correct",
                    help="trace_correct: only rescue pairs where trace majority == gold and final != gold (clean signal). "
                         "any_wrong: rescue any pair where final != gold (uses raw gold answer; noisier).")
    ap.add_argument("--max_per_family", type=int, default=2000,
                    help="cap per gold family to avoid class imbalance")
    ap.add_argument("--include_correct_demos", type=int, default=0,
                    help="also include up to N already-correct (trace==gold==final) records "
                         "to keep the model's good behaviour stable.")
    ap.add_argument("--rescue_weight", type=float, default=1.5,
                    help="sample_weight for rescue records")
    ap.add_argument("--demo_weight", type=float, default=0.5,
                    help="sample_weight for correct demonstrations")
    args = ap.parse_args()

    assert len(args.prompt_jsonl) == len(args.pred_jsonl), \
        "pair --prompt_jsonl and --pred_jsonl 1:1"

    labels = _load_labels()
    rescues = []
    demos = []
    n_seen = 0
    n_skip_no_gold = 0
    n_skip_no_prompt = 0
    for pp, sp in zip(args.prompt_jsonl, args.pred_jsonl):
        prompts = _load_prompts(pp)
        with open(sp) as f:
            for line in f:
                r = json.loads(line)
                if (r.get("input_order") or "ab") != "ab":
                    continue
                pid = r["pair_id"]
                n_seen += 1
                lbl = labels.get(pid)
                if not lbl:
                    n_skip_no_gold += 1
                    continue
                if pid not in prompts:
                    n_skip_no_prompt += 1
                    continue
                gold = lbl["family"]
                fp = r.get("final_prediction") or {}
                fam = fp.get("family")
                if not fp.get("abstain", False) and fam == gold:
                    if args.include_correct_demos:
                        rec = build_record(prompts[pid], r, lbl)
                        rec["sample_weight"] = args.demo_weight
                        rec["tier"] = "trace_align_demo"
                        demos.append(rec)
                    continue
                tm = _trace_majority(r.get("trace"))
                if args.mode == "trace_correct":
                    if tm != gold:
                        continue
                rec = build_record(prompts[pid], r, lbl)
                rec["sample_weight"] = args.rescue_weight
                rescues.append(rec)

    if args.max_per_family:
        cap_buckets = {f: [] for f in FAMS}
        for r in rescues:
            cap_buckets.setdefault(r["family"], []).append(r)
        rescues = []
        for f, recs in cap_buckets.items():
            rescues.extend(recs[: args.max_per_family])

    if args.include_correct_demos > 0:
        cap_buckets = {f: [] for f in FAMS}
        for r in demos:
            cap_buckets.setdefault(r["family"], []).append(r)
        per_fam = max(args.include_correct_demos // max(len(FAMS), 1), 1)
        demos = []
        for f, recs in cap_buckets.items():
            demos.extend(recs[:per_fam])

    final = rescues + demos
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_fam = Counter(r["family"] for r in rescues)
    print(f"[trace-align] seen={n_seen} no_gold={n_skip_no_gold} no_prompt={n_skip_no_prompt}")
    print(f"[trace-align] rescues={len(rescues)} demos={len(demos)} total={len(final)}")
    print(f"[trace-align] per-family rescues: {dict(by_fam)}")
    print(f"[trace-align] wrote {args.output}")


if __name__ == "__main__":
    main()
