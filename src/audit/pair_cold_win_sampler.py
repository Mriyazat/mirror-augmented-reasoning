"""Sample qualitative pair-cold wins/losses for paper examples.

Produces a CSV with:
  - LLM wins vs baseline
  - baseline wins vs LLM
  - both wrong
  - both correct

Stratified by family with deterministic sampling.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def _manifest(path: Path) -> set[str]:
    out = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["pair_id"])
    return out


def _load_truth(keep: set[str]):
    rows = pq.read_table(
        ROOT / "data_processed" / "labels_hierarchical.parquet",
        columns=["pair_id", "family", "subtype", "description"],
    ).to_pylist()
    return {r["pair_id"]: r for r in rows if r["pair_id"] in keep}


def _load_preds(path: Path, keep: set[str]):
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("pair_id")
            if pid not in keep:
                continue
            if (rec.get("input_order") or "ab").lower() != "ab":
                continue
            if pid in out:
                continue
            fp = rec.get("final_prediction") or {}
            out[pid] = {
                "family": fp.get("family"),
                "abstain": bool(fp.get("abstain", False)),
                "confidence": fp.get("confidence"),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_jsonl", required=True)
    ap.add_argument("--llm_predictions", required=True)
    ap.add_argument("--baseline_predictions", required=True)
    ap.add_argument("--llm_name", default="LLM")
    ap.add_argument("--baseline_name", default="baseline")
    ap.add_argument("--per_bucket_per_family", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    keep = _manifest(Path(args.manifest_jsonl))
    truth = _load_truth(keep)
    llm = _load_preds(Path(args.llm_predictions), keep)
    base = _load_preds(Path(args.baseline_predictions), keep)
    common = sorted(set(truth) & set(llm) & set(base))
    if not common:
        raise SystemExit("[sampler] no overlap among truth/llm/baseline.")

    rng = np.random.default_rng(args.seed)

    buckets = defaultdict(list)
    for pid in common:
        g = truth[pid]["family"]
        p_l = llm[pid]["family"] if not llm[pid]["abstain"] else "__abstain__"
        p_b = base[pid]["family"] if not base[pid]["abstain"] else "__abstain__"
        c_l = int(p_l == g)
        c_b = int(p_b == g)
        if c_l == 1 and c_b == 0:
            bucket = "llm_win"
        elif c_l == 0 and c_b == 1:
            bucket = "baseline_win"
        elif c_l == 0 and c_b == 0:
            bucket = "both_wrong"
        else:
            bucket = "both_correct"
        buckets[(bucket, g)].append(
            {
                "pair_id": pid,
                "gold_family": g,
                "gold_subtype": truth[pid].get("subtype"),
                "description": truth[pid].get("description"),
                "llm_pred_family": p_l,
                "baseline_pred_family": p_b,
                "llm_confidence": llm[pid]["confidence"],
                "baseline_confidence": base[pid]["confidence"],
                "bucket": bucket,
            }
        )

    rows = []
    for key, items in buckets.items():
        k = min(args.per_bucket_per_family, len(items))
        if k == 0:
            continue
        idx = rng.choice(len(items), size=k, replace=False)
        rows.extend(items[i] for i in idx)

    # stable sort for readability
    rows.sort(key=lambda r: (r["bucket"], r["gold_family"], r["pair_id"]))

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "bucket",
                "gold_family",
                "gold_subtype",
                "llm_pred_family",
                "baseline_pred_family",
                "llm_confidence",
                "baseline_confidence",
                "description",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(
        f"[sampler] common={len(common):,} sampled={len(rows):,} "
        f"(per_bucket_per_family={args.per_bucket_per_family})"
    )
    print(f"[sampler] wrote {out}")


if __name__ == "__main__":
    main()
