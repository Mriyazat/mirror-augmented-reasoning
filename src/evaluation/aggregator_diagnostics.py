"""Diagnose the new aggregators: per-family F1, abstention distribution,
and mirror family stability (MFS).

Run from DDI/ root.  Operates on the JSONL files produced by
src.inference.aggregate_rerank (which itself reads predict_with_rerank
output)."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]


def _load_predictions(path: Path) -> dict[str, dict]:
    """pair_id -> {ab_family, ba_family, ab_abstain, ba_abstain}."""
    out: dict[str, dict] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("pair_id")
            if not pid:
                continue
            order = (rec.get("input_order") or "ab").lower()
            fp = rec.get("final_prediction") or {}
            out[pid][f"{order}_family"]  = fp.get("family")
            out[pid][f"{order}_abstain"] = bool(fp.get("abstain", False))
    return dict(out)


def _load_truth(path: Path, keep: set[str] | None) -> dict[str, str]:
    rows = pq.read_table(path, columns=["pair_id", "family"]).to_pylist()
    out = {r["pair_id"]: r["family"] for r in rows}
    if keep is not None:
        out = {pid: f for pid, f in out.items() if pid in keep}
    return out


def _manifest(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    pids: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pids.add(json.loads(line)["pair_id"])
    return pids


def per_family(preds, truth, drop_abstain: bool) -> tuple[dict, float]:
    """Returns per_family_f1 dict + macro-F1.  Uses AB ordering only.
    drop_abstain=True excludes abstained records from scoring."""
    labels = sorted(set(truth.values()))
    lab2idx = {f: i for i, f in enumerate(labels)}
    sentinel = len(labels)
    yt, yp = [], []
    for pid, rec in preds.items():
        gold = truth.get(pid)
        if gold is None:
            continue
        if "ab_family" not in rec:
            continue
        if drop_abstain and rec.get("ab_abstain", False):
            continue
        yt.append(lab2idx[gold])
        pred = rec["ab_family"]
        yp.append(lab2idx.get(pred, sentinel))
    if not yt:
        return {}, 0.0
    yt = np.asarray(yt); yp = np.asarray(yp)
    per = {}
    for cls, fam in enumerate(labels):
        yb_t = (yt == cls).astype(int)
        yb_p = (yp == cls).astype(int)
        per[fam] = {
            "f1":      float(f1_score(yb_t, yb_p, zero_division=0)),
            "support": int(yb_t.sum()),
        }
    macro = float(f1_score(yt, yp, labels=list(range(len(labels))),
                           average="macro", zero_division=0))
    return per, macro


def abstention_by_family(preds, truth) -> dict:
    """For variants that may abstain, count abstentions per GOLD family."""
    abst_by_gold: Counter = Counter()
    total_by_gold: Counter = Counter()
    for pid, rec in preds.items():
        gold = truth.get(pid)
        if gold is None or "ab_family" not in rec:
            continue
        total_by_gold[gold] += 1
        if rec.get("ab_abstain", False):
            abst_by_gold[gold] += 1
    return {fam: {"abst": int(abst_by_gold[fam]),
                  "total": int(total_by_gold[fam]),
                  "abst_rate": float(abst_by_gold[fam] / max(total_by_gold[fam], 1))}
            for fam in sorted(total_by_gold)}


def mfs(preds) -> dict:
    """Mirror Family Stability: fraction of pairs where the AB and BA
    decisions match on family.  Skips pairs that lack one of the two
    orderings.  For variants that may abstain, computes two numbers:
      mfs_strict   -- both ABstain flags must be False AND families match
      mfs_loose    -- families match (treating abstain as match-anything)
    """
    n_total = n_have_both = 0
    n_match = 0
    n_abst_either = 0
    for pid, rec in preds.items():
        if "ab_family" not in rec or "ba_family" not in rec:
            continue
        n_have_both += 1
        ab_abst = rec.get("ab_abstain", False)
        ba_abst = rec.get("ba_abstain", False)
        if ab_abst or ba_abst:
            n_abst_either += 1
            continue
        n_total += 1
        if rec["ab_family"] == rec["ba_family"]:
            n_match += 1
    return {
        "n_have_both":      int(n_have_both),
        "n_scored":         int(n_total),
        "n_abst_either":    int(n_abst_either),
        "mfs":              float(n_match / max(n_total, 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="Aggregator output JSONL (both AB+BA, merged).")
    ap.add_argument("--labels", default="data_processed/labels_hierarchical.parquet")
    ap.add_argument("--manifest_jsonl", default=None)
    ap.add_argument("--drop_abstain", action="store_true",
                    help="Compute per-family F1 in the drop-abstain mode "
                         "(selective). Default scores abstained = keep.")
    ap.add_argument("--variant_label", default="(variant)")
    args = ap.parse_args()

    keep = _manifest(Path(args.manifest_jsonl) if args.manifest_jsonl else None)
    truth = _load_truth(Path(args.labels), keep)
    preds = _load_predictions(Path(args.predictions))
    if keep is not None:
        preds = {pid: r for pid, r in preds.items() if pid in keep}

    per, macro = per_family(preds, truth, args.drop_abstain)
    abst = abstention_by_family(preds, truth)
    mfs_res = mfs(preds)

    print(f"\n==== {args.variant_label}  ({Path(args.predictions).name}) ====")
    print(f"   macro-F1 (drop_abstain={args.drop_abstain}): {macro:.4f}")
    print(f"   MFS (AB==BA family, both committed): {mfs_res['mfs']:.4f}  "
          f"(scored={mfs_res['n_scored']:,}, abstain-either={mfs_res['n_abst_either']:,})")
    print(f"\n   Per-family F1:")
    print(f"     {'family':22s}  {'F1':>6s}  {'support':>8s}  "
          f"{'abst':>6s}  {'abst%':>7s}")
    for fam in sorted(per.keys()):
        pf = per[fam]
        ab = abst.get(fam, {"abst": 0, "total": 0, "abst_rate": 0.0})
        print(f"     {fam:22s}  {pf['f1']:.4f}  {pf['support']:8,d}  "
              f"{ab['abst']:6,d}  {ab['abst_rate']*100:6.1f}%")
    # Rare-only macro
    rare = ["PK_Absorption", "PK_Distribution", "PK_Excretion", "Efficacy"]
    rare_f1 = [per[f]["f1"] for f in rare if f in per]
    if rare_f1:
        print(f"\n   Rare-only macro-F1 (PK_Absorption,Distribution,Excretion,Efficacy): "
              f"{np.mean(rare_f1):.4f}")


if __name__ == "__main__":
    main()
