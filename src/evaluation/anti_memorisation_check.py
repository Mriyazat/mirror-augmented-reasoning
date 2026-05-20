"""Quick: Spearman rho of per-pair accuracy vs training-pair coverage of
the rarer endpoint, for an arbitrary prediction file.

The previous leakage_probe.py was more elaborate (per-decile breakdown
+ figure).  This is a one-shot rho number for any prediction file so we
can confirm "the new winners preserve the anti-memorisation property."
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", default="random_full",
                    choices=["random_full", "drug_cold", "pair_cold"])
    ap.add_argument("--manifest_jsonl", default=None)
    ap.add_argument("--drop_abstain", action="store_true")
    args = ap.parse_args()

    # 1) per-drug training coverage for this split
    manifest = pq.read_table(
        DATA / "splits" / f"manifest_{args.split}.parquet"
    ).to_pylist()
    train_pids = {r["pair_id"] for r in manifest if r["split"] == "train"}
    pairs = pq.read_table(
        DATA / "pairs.parquet", columns=["pair_id", "a_id", "b_id"]
    ).to_pylist()
    pair_idx = {r["pair_id"]: r for r in pairs}
    drug_freq: Counter = Counter()
    for pid in train_pids:
        r = pair_idx.get(pid)
        if not r:
            continue
        drug_freq[r["a_id"]] += 1
        drug_freq[r["b_id"]] += 1

    # 2) gold + manifest filter
    truth = {r["pair_id"]: r["family"] for r in
             pq.read_table(DATA / "labels_hierarchical.parquet",
                           columns=["pair_id", "family"]).to_pylist()}
    keep = None
    if args.manifest_jsonl:
        keep = set()
        with open(args.manifest_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    keep.add(json.loads(line)["pair_id"])

    # 3) per-pair record
    freqs = []
    correct = []
    with open(args.predictions) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("pair_id")
            if not pid or (keep is not None and pid not in keep):
                continue
            if (rec.get("input_order") or "ab") != "ab":
                continue
            fp = rec.get("final_prediction") or {}
            if args.drop_abstain and fp.get("abstain"):
                continue
            gold = truth.get(pid)
            if gold is None:
                continue
            ep = pair_idx.get(pid)
            if not ep:
                continue
            fmin = min(drug_freq.get(ep["a_id"], 0), drug_freq.get(ep["b_id"], 0))
            freqs.append(fmin)
            correct.append(1 if fp.get("family") == gold else 0)

    if not freqs:
        raise SystemExit("[mem] no rows.")
    freqs = np.asarray(freqs); correct = np.asarray(correct)

    # Spearman rho (manual)
    rx = np.argsort(np.argsort(freqs))
    ry = np.argsort(np.argsort(correct))
    n = len(freqs)
    rho = 1.0 - 6.0 * float(np.sum((rx - ry) ** 2)) / (n * (n * n - 1))

    # Top/bottom decile accuracy
    order = np.argsort(freqs, kind="stable")
    deciles = [int(i * 10 / n) for i in range(n)]
    bot = correct[order][:n // 10].mean() if n >= 10 else float("nan")
    top = correct[order][-n // 10:].mean() if n >= 10 else float("nan")
    print(f"[mem] {n:,} pairs  acc={correct.mean():.4f}  "
          f"Spearman_rho(freq_min vs correct) = {rho:+.4f}  "
          f"bottom_decile_acc = {bot:.4f}  top_decile_acc = {top:.4f}  "
          f"delta = {top-bot:+.4f}")


if __name__ == "__main__":
    main()
