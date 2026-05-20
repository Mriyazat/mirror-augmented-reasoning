"""Per-class F1 with 95% bootstrap CIs.

For one prediction file, emit a table:
    family | n_gold | n_pred | F1 [95% CI] | Precision [CI] | Recall [CI]

Abstentions are scored as a sentinel class so they cost recall on the
true family but never artificially inflate precision elsewhere -- the
same convention as run_full_eval.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_fscore_support


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


def _load_predictions(path: Path, drop_abstain: bool, use_ab_only: bool):
    """Yield (pair_id, pred_family_or_None_for_abstain)."""
    seen = set()
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
            if use_ab_only and order != "ab":
                continue
            if not use_ab_only and pid in seen:
                continue
            seen.add(pid)
            fp = rec.get("final_prediction") or {}
            if fp.get("abstain"):
                if drop_abstain:
                    continue
                pred = None
            else:
                pred = fp.get("family")
            yield pid, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--labels", default="data_processed/labels_hierarchical.parquet")
    ap.add_argument("--manifest_jsonl", default=None)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--drop_abstain", action="store_true")
    ap.add_argument("--use_ab_only", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    keep = _manifest(Path(args.manifest_jsonl) if args.manifest_jsonl else None)
    truth = _load_truth(Path(args.labels), keep)
    if not truth:
        raise SystemExit("[bs] no truth rows after filter")

    pred_map: dict[str, str | None] = {}
    for pid, pred in _load_predictions(Path(args.predictions),
                                       args.drop_abstain,
                                       args.use_ab_only):
        if pid in truth and pid not in pred_map:
            pred_map[pid] = pred

    pids = sorted(pred_map.keys() & truth.keys())
    if not pids:
        raise SystemExit("[bs] no overlap between predictions and labels")

    labels = sorted(set(truth.values()))
    lab2idx = {f: i for i, f in enumerate(labels)}
    SENTINEL = len(labels)  # abstain bucket

    yt = np.asarray([lab2idx[truth[pid]] for pid in pids])
    yp = np.asarray([SENTINEL if pred_map[pid] is None
                     else lab2idx.get(pred_map[pid], SENTINEL) for pid in pids])

    print(f"[bs] run={args.run_name}  n_pairs={len(pids)}  n_classes={len(labels)}")
    print(f"[bs] abstain rate: {(yp == SENTINEL).mean():.4f}")

    n = len(pids)
    rng = np.random.default_rng(args.seed)
    boot_pf = np.zeros((args.n_boot, len(labels)), dtype=float)
    boot_pp = np.zeros((args.n_boot, len(labels)), dtype=float)
    boot_pr = np.zeros((args.n_boot, len(labels)), dtype=float)
    for b in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        p, r, f1, _ = precision_recall_fscore_support(
            yt[idx], yp[idx], labels=list(range(len(labels))),
            average=None, zero_division=0,
        )
        boot_pf[b] = f1
        boot_pp[b] = p
        boot_pr[b] = r
    # Point estimates (no bootstrap)
    p_pt, r_pt, f1_pt, sup = precision_recall_fscore_support(
        yt, yp, labels=list(range(len(labels))),
        average=None, zero_division=0,
    )
    n_pred_per = np.zeros(len(labels), dtype=int)
    for v in yp:
        if v < len(labels):
            n_pred_per[v] += 1

    print(f"\n  {'family':18s}  {'n_gold':>6s}  {'n_pred':>6s}  "
          f"{'F1':>16s}  {'Precision':>16s}  {'Recall':>16s}")
    for i, fam in enumerate(labels):
        f1_lo, f1_hi = np.percentile(boot_pf[:, i], [2.5, 97.5])
        p_lo, p_hi = np.percentile(boot_pp[:, i], [2.5, 97.5])
        r_lo, r_hi = np.percentile(boot_pr[:, i], [2.5, 97.5])
        print(f"  {fam:18s}  {int(sup[i]):6d}  {int(n_pred_per[i]):6d}  "
              f"{f1_pt[i]:.3f} [{f1_lo:.3f},{f1_hi:.3f}]  "
              f"{p_pt[i]:.3f} [{p_lo:.3f},{p_hi:.3f}]  "
              f"{r_pt[i]:.3f} [{r_lo:.3f},{r_hi:.3f}]")
    macro_f1 = f1_pt.mean()
    boot_macro = boot_pf.mean(axis=1)
    macro_lo, macro_hi = np.percentile(boot_macro, [2.5, 97.5])
    print(f"\n  macro-F1 = {macro_f1:.4f}  95% CI [{macro_lo:.4f}, {macro_hi:.4f}]")

    if args.output:
        out_rows = [{
            "run":        args.run_name,
            "family":     fam,
            "n_gold":     int(sup[i]),
            "n_pred":     int(n_pred_per[i]),
            "f1":         float(f1_pt[i]),
            "f1_lo":      float(np.percentile(boot_pf[:, i], 2.5)),
            "f1_hi":      float(np.percentile(boot_pf[:, i], 97.5)),
            "precision":  float(p_pt[i]),
            "precision_lo": float(np.percentile(boot_pp[:, i], 2.5)),
            "precision_hi": float(np.percentile(boot_pp[:, i], 97.5)),
            "recall":     float(r_pt[i]),
            "recall_lo":  float(np.percentile(boot_pr[:, i], 2.5)),
            "recall_hi":  float(np.percentile(boot_pr[:, i], 97.5)),
        } for i, fam in enumerate(labels)]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        print(f"[bs] wrote {args.output}")


if __name__ == "__main__":
    main()
