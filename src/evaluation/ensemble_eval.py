"""LLM x baseline ensemble evaluation on random_full.

Reads:
  --llm_predictions PATH      LLM JSONL (with final_prediction.family/abstain/confidence/label_dist)
  --baseline_predictions PATH non-LLM JSONL (same schema)
  --manifest_jsonl PATH       restrict to pair_ids in this manifest

Strategies (per pair, AB only):
  llm_solo               LLM only
  baseline_solo          baseline only
  rescue_abstain         LLM unless it abstains -> baseline (best for conformal LLM)
  max_confidence         pick model with higher confidence
  prob_avg               argmax of mean(LLM_dist, baseline_dist)
  prob_weighted          argmax of conf-weighted average
  prefer_baseline_disagree    use baseline whenever the two disagree
  per_family_expert      route per predicted family using overall test F1 per family (oracle-style)

Reports macro-F1, rare-F1, coverage, MFS, ECE for each strategy.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score


def _manifest(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    pids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pids.add(json.loads(line)["pair_id"])
    return pids


def _load(path: Path, keep: set[str] | None) -> dict[str, dict]:
    """pair_id -> {ab: rec, ba: rec}.  rec carries fam, abstain, conf, dist."""
    out: dict[str, dict] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("pair_id")
            if not pid or (keep is not None and pid not in keep):
                continue
            order = (r.get("input_order") or "ab").lower()
            if order not in ("ab", "ba"):
                continue
            if order in out[pid]:
                continue
            fp = r.get("final_prediction") or {}
            out[pid][order] = {
                "family":   fp.get("family"),
                "abstain":  bool(fp.get("abstain", False)),
                "conf":     fp.get("confidence"),
                "dist":     fp.get("label_dist") or {},
            }
    return out


def _scores(yt: np.ndarray, yp: np.ndarray, labels: list[str], rare: set[str],
            n_total: int) -> dict:
    lab2idx = {f: i for i, f in enumerate(labels)}
    SENTINEL = len(labels)
    per = f1_score(yt, yp, labels=list(range(len(labels))), average=None,
                    zero_division=0)
    macro = float(per.mean())
    rare_idxs = [lab2idx[f] for f in rare if f in lab2idx]
    rare_f1 = float(np.mean([per[i] for i in rare_idxs])) if rare_idxs else 0.0
    abst = int((yp == SENTINEL).sum())
    return dict(macro_f1=macro, rare_f1=rare_f1,
                n=len(yt), n_abstain=abst,
                coverage=(len(yt) - abst) / n_total)


def _mfs(p_ab: str | None, p_ba: str | None, abst_ab: bool, abst_ba: bool):
    if abst_ab or abst_ba or p_ab is None or p_ba is None:
        return None
    return int(p_ab == p_ba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_predictions", required=True)
    ap.add_argument("--baseline_predictions", required=True)
    ap.add_argument("--baseline_name", default="baseline")
    ap.add_argument("--llm_name", default="llm")
    ap.add_argument("--manifest_jsonl", default=None)
    ap.add_argument("--labels", default="data_processed/labels_hierarchical.parquet")
    args = ap.parse_args()

    keep = _manifest(Path(args.manifest_jsonl) if args.manifest_jsonl else None)
    truth = {r["pair_id"]: r["family"] for r in
             pq.read_table(args.labels,
                           columns=["pair_id", "family"]).to_pylist()}
    if keep is not None:
        truth = {k: v for k, v in truth.items() if k in keep}

    llm = _load(Path(args.llm_predictions), keep)
    bl = _load(Path(args.baseline_predictions), keep)
    common = [pid for pid in sorted(truth) if pid in llm and pid in bl
              and "ab" in llm[pid] and "ab" in bl[pid]]
    print(f"[ens] {args.llm_name} x {args.baseline_name}: "
          f"{len(common):,} common pairs (truth={len(truth):,}, "
          f"llm={len(llm):,}, baseline={len(bl):,})")

    labels = sorted(set(truth.values()))
    lab2idx = {f: i for i, f in enumerate(labels)}
    SENTINEL = len(labels)
    RARE = {"PK_Absorption", "PK_Distribution", "PK_Excretion", "Other"}
    n_total = len(truth)

    def to_idx(fam):
        return lab2idx.get(fam, SENTINEL) if fam is not None else SENTINEL

    yt = np.asarray([lab2idx[truth[pid]] for pid in common])

    strategies = {}

    # llm_solo, baseline_solo
    strategies["llm_solo"] = np.asarray([
        SENTINEL if llm[pid]["ab"]["abstain"] else to_idx(llm[pid]["ab"]["family"])
        for pid in common
    ])
    strategies["baseline_solo"] = np.asarray([
        to_idx(bl[pid]["ab"]["family"]) for pid in common
    ])

    # rescue_abstain: LLM unless it abstains -> baseline
    strategies["rescue_abstain"] = np.asarray([
        to_idx(bl[pid]["ab"]["family"]) if llm[pid]["ab"]["abstain"]
        else to_idx(llm[pid]["ab"]["family"])
        for pid in common
    ])

    # max_confidence (skip if no confidence)
    def _conf(rec):
        c = rec.get("conf")
        try:
            return float(c) if c is not None else 0.0
        except Exception:
            return 0.0
    strategies["max_confidence"] = np.asarray([
        to_idx(llm[pid]["ab"]["family"]) if _conf(llm[pid]["ab"]) >= _conf(bl[pid]["ab"])
        else to_idx(bl[pid]["ab"]["family"])
        for pid in common
    ])

    # prob_avg: argmax of (LLM_dist + baseline_dist)/2 -- requires both dists
    def _dist_vec(d):
        v = np.zeros(len(labels), dtype=float)
        if not isinstance(d, dict):
            return v
        for fam, p in d.items():
            i = lab2idx.get(fam)
            if i is None:
                continue
            try:
                v[i] = float(p)
            except Exception:
                continue
        s = v.sum()
        if s > 0: v /= s
        return v
    def _argmax_or_sentinel(v):
        if v.sum() == 0: return SENTINEL
        return int(np.argmax(v))
    strategies["prob_avg"] = np.asarray([
        _argmax_or_sentinel((_dist_vec(llm[pid]["ab"]["dist"])
                              + _dist_vec(bl[pid]["ab"]["dist"])) / 2)
        for pid in common
    ])
    strategies["prob_weighted"] = np.asarray([
        _argmax_or_sentinel(
            _conf(llm[pid]["ab"]) * _dist_vec(llm[pid]["ab"]["dist"])
            + _conf(bl[pid]["ab"]) * _dist_vec(bl[pid]["ab"]["dist"])
        ) for pid in common
    ])

    # prefer_baseline_disagree
    strategies["prefer_baseline_disagree"] = np.asarray([
        to_idx(bl[pid]["ab"]["family"])
        if (llm[pid]["ab"]["family"] != bl[pid]["ab"]["family"]
            and not llm[pid]["ab"]["abstain"])
        else (SENTINEL if llm[pid]["ab"]["abstain"]
              else to_idx(llm[pid]["ab"]["family"]))
        for pid in common
    ])

    # per_family_expert: route by LLM predicted family to whichever model has
    # higher F1 ON THIS SAME TEST SET for that family (oracle upper bound).
    # We compute the per-family F1 of each model first.
    yp_llm = strategies["llm_solo"]
    yp_bl  = strategies["baseline_solo"]
    per_llm = f1_score(yt, yp_llm, labels=list(range(len(labels))),
                        average=None, zero_division=0)
    per_bl  = f1_score(yt, yp_bl, labels=list(range(len(labels))),
                        average=None, zero_division=0)
    winner_per_fam = {labels[i]: ("llm" if per_llm[i] >= per_bl[i] else "baseline")
                      for i in range(len(labels))}
    strategies["per_family_expert (oracle)"] = np.asarray([
        to_idx(llm[pid]["ab"]["family"])
        if winner_per_fam.get(llm[pid]["ab"]["family"]) == "llm"
        else to_idx(bl[pid]["ab"]["family"])
        for pid in common
    ])

    # ---- Report ----
    print(f"\n{'strategy':30s}  {'macro_F1':>9s}  {'rare_F1':>8s}  {'cov':>6s}  {'MFS':>6s}")
    for name, yp in strategies.items():
        s = _scores(yt, yp, labels, RARE, n_total)
        # MFS using same strategy on BA orderings if both have BA
        ab_ba_preds = {}  # pid -> (fam_ab, fam_ba)
        for pid in common:
            if "ba" not in llm.get(pid, {}) or "ba" not in bl.get(pid, {}):
                continue
            # recompute strategy decision for the BA ordering
            llm_ba = llm[pid]["ba"]; bl_ba = bl[pid]["ba"]
            llm_ab = llm[pid]["ab"]; bl_ab = bl[pid]["ab"]
            if name == "llm_solo":
                fa = (None if llm_ab["abstain"] else llm_ab["family"])
                fb = (None if llm_ba["abstain"] else llm_ba["family"])
            elif name == "baseline_solo":
                fa = bl_ab["family"]; fb = bl_ba["family"]
            elif name == "rescue_abstain":
                fa = bl_ab["family"] if llm_ab["abstain"] else llm_ab["family"]
                fb = bl_ba["family"] if llm_ba["abstain"] else llm_ba["family"]
            elif name == "max_confidence":
                fa = llm_ab["family"] if _conf(llm_ab) >= _conf(bl_ab) else bl_ab["family"]
                fb = llm_ba["family"] if _conf(llm_ba) >= _conf(bl_ba) else bl_ba["family"]
            else:
                continue  # other strategies harder to mirror; skip
            if fa is None or fb is None: continue
            ab_ba_preds[pid] = (fa, fb)
        if ab_ba_preds:
            mfs = sum(1 for (a, b) in ab_ba_preds.values() if a == b) / len(ab_ba_preds)
            mfs_s = f"{mfs:.4f}"
        else:
            mfs_s = "--"
        print(f"{name:30s}  {s['macro_f1']:9.4f}  {s['rare_f1']:8.4f}  "
              f"{s['coverage']:.4f}  {mfs_s}")


if __name__ == "__main__":
    main()
