"""Compare confidence signals as selective-prediction abstention scores.

For each test pair, we compute several candidate confidence signals
(PRM, vote-margin, vote-entropy, LLM self-confidence) and the gold
correctness of the aggregator's prediction.  Then for each signal we
report:
  - AUROC vs correctness  (higher = better selection signal)
  - selective F1 at coverage = 0.6  (apples-to-apples comparison)
  - Spearman rho between the signal and the per-pair correctness

If vote-margin's AUROC is comparable to PRM's, we can drop the PRM
entirely at inference time and use vote-margin as the conformal score
-- a 70B-parameter saving with the same accuracy/coverage trade-off.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score, f1_score

ROOT = Path(__file__).resolve().parents[2]


def _load_rerank_file(path: Path):
    """Yields one record per pair (AB only) with full candidates."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if (rec.get("input_order") or "ab") != "ab":
                continue
            yield rec


def _load_truth(labels_path: Path, keep: set[str] | None) -> dict[str, str]:
    rows = pq.read_table(labels_path, columns=["pair_id", "family"]).to_pylist()
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


def _candidate_signals(cands: list[dict], metric_key: str = "prm_geomean"):
    """Compute every per-pair confidence signal from the N candidates."""
    fams = []
    prms = []
    for c in cands:
        if not c.get("parse_ok"):
            continue
        f = c.get("family")
        if not f:
            continue
        try:
            p = float(c.get(metric_key) or 0.0)
        except Exception:
            p = 0.0
        fams.append(f)
        prms.append(p)
    if not fams:
        return None
    n = len(fams)
    counter = Counter(fams)
    most = counter.most_common()
    top_fam, top_n = most[0]
    second_n = most[1][1] if len(most) > 1 else 0
    # vote margin in [0, 1]
    vote_margin = (top_n - second_n) / n
    # vote entropy (Shannon, nats)
    H = 0.0
    for _, c in most:
        p = c / n
        if p > 0:
            H -= p * math.log(p)
    # max possible H = log(min(n_families, 7))
    H_norm = H / math.log(min(n, 7)) if n > 1 else 0.0
    # PRM aggregates
    prm_arr = np.asarray(prms, dtype=float)
    prm_top = float(prm_arr.max())
    prm_mean = float(prm_arr.mean())
    # PRM of the top-vote family only
    top_fam_prms = [p for p, f in zip(prms, fams) if f == top_fam]
    prm_top_fam_mean = float(np.mean(top_fam_prms)) if top_fam_prms else 0.0
    return {
        "top_fam":          top_fam,
        "top_vote_n":       top_n,
        "n_cands":          n,
        "vote_margin":      vote_margin,
        "vote_entropy_neg": -H_norm,  # negate so "high = confident"
        "prm_max":          prm_top,
        "prm_mean":         prm_mean,
        "prm_top_fam_mean": prm_top_fam_mean,
    }


def _selective_f1(yt: np.ndarray, yp: np.ndarray, scores: np.ndarray,
                  labels_n: int, coverage: float) -> tuple[float, int]:
    """Sort by score descending, keep top `coverage` fraction, compute macro-F1."""
    n = len(yt)
    keep_n = int(coverage * n)
    if keep_n < 10:
        return 0.0, keep_n
    order = np.argsort(-scores, kind="stable")
    idx = order[:keep_n]
    return float(f1_score(yt[idx], yp[idx], labels=list(range(labels_n)),
                          average="macro", zero_division=0)), keep_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank_input", required=True,
                    help="Path to a predict_with_rerank JSONL with `rerank.candidates`.")
    ap.add_argument("--labels", default="data_processed/labels_hierarchical.parquet")
    ap.add_argument("--manifest_jsonl", default=None)
    ap.add_argument("--coverage_targets", default="0.95,0.80,0.60,0.40")
    args = ap.parse_args()

    keep = _manifest(Path(args.manifest_jsonl) if args.manifest_jsonl else None)
    truth = _load_truth(Path(args.labels), keep)

    # Collect per-pair signals + correctness (using vote_majority as the prediction)
    rows = []
    for rec in _load_rerank_file(Path(args.rerank_input)):
        pid = rec.get("pair_id")
        gold = truth.get(pid)
        if gold is None:
            continue
        cands = (rec.get("rerank") or {}).get("candidates") or []
        sig = _candidate_signals(cands)
        if sig is None:
            continue
        # Use vote_majority family as the prediction
        correct = int(sig["top_fam"] == gold)
        rows.append({
            "pair_id":      pid,
            "gold":         gold,
            "pred":         sig["top_fam"],
            "correct":      correct,
            **{k: v for k, v in sig.items() if isinstance(v, (int, float))},
        })

    if not rows:
        raise SystemExit("[auroc] no rows after filtering.")
    print(f"[auroc] scored {len(rows):,} pairs (AB only)")

    y_correct = np.asarray([r["correct"] for r in rows])
    base_acc = float(y_correct.mean())
    print(f"[auroc] vote_majority base accuracy (commit-all): {base_acc:.4f}")

    signals = ["prm_max", "prm_mean", "prm_top_fam_mean",
               "vote_margin", "vote_entropy_neg"]

    print(f"\n{'signal':22s}  {'AUROC':>7s}  {'Spearman_rho':>12s}")
    for s in signals:
        scores = np.asarray([r[s] for r in rows], dtype=float)
        if np.all(scores == scores[0]) or len(np.unique(scores)) < 3:
            auroc = float("nan")
        else:
            auroc = float(roc_auc_score(y_correct, scores))
        # Spearman (manual, no scipy.stats dep)
        rx = np.argsort(np.argsort(scores))
        ry = np.argsort(np.argsort(y_correct))
        n = len(scores)
        rho = 1.0 - 6.0 * float(np.sum((rx - ry) ** 2)) / (n * (n * n - 1))
        print(f"{s:22s}  {auroc:7.4f}  {rho:12.4f}")

    # Selective-F1 at multiple coverage targets
    labels_set = sorted(set(truth.values()))
    lab2idx = {f: i for i, f in enumerate(labels_set)}
    sentinel = len(labels_set)
    yt = np.asarray([lab2idx[r["gold"]] for r in rows])
    yp = np.asarray([lab2idx.get(r["pred"], sentinel) for r in rows])

    targets = [float(c) for c in args.coverage_targets.split(",") if c.strip()]
    print(f"\nSelective macro-F1 by signal x coverage target:")
    print(f"  base (no abstain, cov=1.0): macro-F1 = "
          f"{f1_score(yt, yp, labels=list(range(len(labels_set))), average='macro', zero_division=0):.4f}")
    print(f"  {'signal':22s}", end="")
    for c in targets:
        print(f"  cov={c:.2f}", end="")
    print()
    for s in signals:
        scores = np.asarray([r[s] for r in rows], dtype=float)
        print(f"  {s:22s}", end="")
        for c in targets:
            f1, n = _selective_f1(yt, yp, scores, len(labels_set), c)
            print(f"   {f1:.4f}", end="")
        print()


if __name__ == "__main__":
    main()
