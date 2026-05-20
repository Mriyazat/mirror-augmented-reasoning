"""Build a family-stratified evaluation manifest from `manifest_<split>.parquet`.

Why
---
Running a 145K-pair test partition through the LLM costs 70+ GPU-hours per
phase, so we evaluate on a 5K stratified subset instead.  Stratifying on
family ensures the rare classes (PK_Distribution, PK_Excretion) get
sampled at a meaningful rate and per-family selective accuracy is not
estimated from <50 examples.

Output
------
A JSONL with one record per line:
    {"pair_id": str, "family": str, "split": "test"}
matching the schema that `build_student_eval_prompts.py` and
`baseline_xgboost.py` already understand.

The `random_full_test_5000_stratified.manifest.jsonl` we already use was
produced by an ad-hoc snippet; this script standardises that recipe so the
same stratification is reproducible for `drug_cold_test` and
`pair_cold_test`.

Usage
-----
    python -m src.data.build_stratified_manifest \
        --split        drug_cold \
        --section      test \
        --n            5000 \
        --output       outputs/eval_prompts/drug_cold_test_5000_stratified.manifest.jsonl

    python -m src.data.build_stratified_manifest \
        --split        pair_cold \
        --section      test \
        --n            5000 \
        --output       outputs/eval_prompts/pair_cold_test_5000_stratified.manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data_processed" / "splits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True,
                    choices=["random_full", "drug_cold", "pair_cold"])
    ap.add_argument("--section", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=5000,
                    help="Total number of records to sample.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=20251005,
                    help="Same seed used for the original random_full 5K subset.")
    ap.add_argument("--mode", default="sqrt_inverse",
                    choices=["sqrt_inverse", "proportional", "balanced"],
                    help="sqrt_inverse: per-family count proportional to "
                         "sqrt(1/n_f) (matches phase-3 sampling); "
                         "proportional: keep family ratios from the section; "
                         "balanced: equal count per family.")
    args = ap.parse_args()

    manifest = pq.read_table(
        SPLITS_DIR / f"manifest_{args.split}.parquet"
    ).to_pylist()
    rows = [r for r in manifest if r.get("split") == args.section]
    if not rows:
        raise SystemExit(
            f"No rows with split={args.section} in manifest_{args.split}.parquet"
        )

    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    fams = sorted(by_fam.keys())
    print(f"[manifest] {args.split}/{args.section}: {len(rows):,} rows, "
          f"{len(fams)} families")
    for f in fams:
        print(f"  {f:20s}  n={len(by_fam[f]):,}")

    # ---- compute per-family quotas ----
    if args.mode == "balanced":
        quota = {f: args.n // len(fams) for f in fams}
    elif args.mode == "proportional":
        total = sum(len(by_fam[f]) for f in fams)
        quota = {f: int(round(args.n * len(by_fam[f]) / total)) for f in fams}
    else:  # sqrt_inverse
        weights = {f: math.sqrt(1.0 / max(len(by_fam[f]), 1)) for f in fams}
        wsum = sum(weights.values())
        quota = {f: int(round(args.n * weights[f] / wsum)) for f in fams}

    # Cap by family availability and patch up rounding error
    for f in fams:
        quota[f] = min(quota[f], len(by_fam[f]))
    delta = args.n - sum(quota.values())
    if delta != 0:
        # adjust on the most-populated families that still have headroom
        order = sorted(fams, key=lambda f: -(len(by_fam[f]) - quota[f]))
        i = 0
        step = 1 if delta > 0 else -1
        while delta != 0 and order:
            f = order[i % len(order)]
            head = len(by_fam[f]) - quota[f]
            if step > 0 and head > 0:
                quota[f] += 1
                delta -= 1
            elif step < 0 and quota[f] > 0:
                quota[f] -= 1
                delta += 1
            i += 1
            if i > 50000:
                break

    print(f"[manifest] mode={args.mode}, total quota = "
          f"{sum(quota.values()):,} of {args.n} requested")
    for f in fams:
        print(f"  {f:20s}  take {quota[f]:,} / {len(by_fam[f]):,}")

    rng = random.Random(args.seed)
    out_rows: list[dict] = []
    for f in fams:
        pool = by_fam[f][:]
        rng.shuffle(pool)
        for r in pool[: quota[f]]:
            out_rows.append({
                "pair_id": r["pair_id"],
                "family":  r["family"],
                "split":   args.section,
            })
    rng.shuffle(out_rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fout:
        for r in out_rows:
            fout.write(json.dumps(r) + "\n")
    print(f"[manifest] wrote {len(out_rows):,} -> {out}")


if __name__ == "__main__":
    main()
