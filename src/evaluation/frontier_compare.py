"""Compare a frontier-LLM prediction file to the student LLM (and baselines)
on the **same set of pair_ids**.

Outputs:
  outputs/diag2/frontier/<split>_<model>.json
  outputs/diag2/frontier/SUMMARY.md (appended)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, restrict: set[str] | None = None) -> dict[str, dict]:
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("pair_id")
            if restrict is not None and pid not in restrict:
                continue
            if (r.get("input_order") or "ab").lower() != "ab":
                continue
            if pid in out:
                continue
            fp = r.get("final_prediction") or {}
            try:
                conf = float(fp.get("confidence")) if fp.get("confidence") is not None else 0.0
            except Exception:
                conf = 0.0
            out[pid] = {
                "family": fp.get("family"),
                "subtype": fp.get("subtype"),
                "abstain": bool(fp.get("abstain", False)),
                "conf": conf,
            }
    return out


def _truth():
    t = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet",
                      columns=["pair_id", "family", "subtype"]).to_pylist()
    return {r["pair_id"]: (r["family"], r["subtype"]) for r in t}


def _metrics(yt, yp, labels):
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_F1": float(f1_score(yt, yp, labels=list(range(len(labels))), average="macro", zero_division=0)),
        "weighted_F1": float(f1_score(yt, yp, labels=list(range(len(labels))), average="weighted", zero_division=0)),
    }


def _to_arrays(pids, truth, preds, lab2idx):
    sentinel = len(lab2idx)
    yt = np.asarray([lab2idx[truth[pid][0]] for pid in pids])
    yp = np.asarray([
        sentinel if (preds[pid]["abstain"] or preds[pid]["family"] not in lab2idx)
        else lab2idx[preds[pid]["family"]]
        for pid in pids
    ])
    return yt, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--mlp", default=None)
    ap.add_argument("--xgb", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model_name", default="gpt-4o")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    truth = _truth()
    frontier = _load(Path(args.frontier))
    student = _load(Path(args.student), restrict=set(frontier))
    mlp = _load(Path(args.mlp), restrict=set(frontier)) if args.mlp else {}
    xgb = _load(Path(args.xgb), restrict=set(frontier)) if args.xgb else {}

    common = sorted(set(frontier) & set(student) & set(truth))
    if not common:
        raise SystemExit("[fc] no overlap.")

    labels = sorted({truth[p][0] for p in common})
    lab2idx = {f: i for i, f in enumerate(labels)}

    yt, yf = _to_arrays(common, truth, frontier, lab2idx)
    _, ys = _to_arrays(common, truth, student, lab2idx)
    rep = {
        "split": args.split,
        "n": len(common),
        "model": args.model_name,
        "frontier": _metrics(yt, yf, labels),
        "student":  _metrics(yt, ys, labels),
    }

    if mlp:
        common_mlp = [pid for pid in common if pid in mlp]
        if common_mlp:
            yt2, ym = _to_arrays(common_mlp, truth, mlp, lab2idx)
            rep["mlp"] = _metrics(yt2, ym, labels) | {"n": len(common_mlp)}
    if xgb:
        common_xgb = [pid for pid in common if pid in xgb]
        if common_xgb:
            yt2, yx = _to_arrays(common_xgb, truth, xgb, lab2idx)
            rep["xgb"] = _metrics(yt2, yx, labels) | {"n": len(common_xgb)}

    # Paired bootstrap: frontier vs student
    def macro(true, pred):
        return f1_score(true, pred, labels=list(range(len(labels))), average="macro", zero_division=0)
    rng = np.random.default_rng(args.seed)
    diffs = np.zeros(args.n_boot)
    n = len(common)
    for b in range(args.n_boot):
        i = rng.integers(0, n, size=n)
        diffs[b] = macro(yt[i], yf[i]) - macro(yt[i], ys[i])
    delta = rep["frontier"]["macro_F1"] - rep["student"]["macro_F1"]
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    p = float(min((diffs <= 0).mean(), (diffs >= 0).mean()) * 2)
    rep["paired_bootstrap_frontier_minus_student"] = {
        "delta_macro_F1": delta,
        "ci95": [lo, hi],
        "p_value": p,
    }

    # Subtype-given-family-correct (LLM only)
    frontier_sub_fam_ok = 0
    frontier_fam_ok = 0
    student_sub_fam_ok = 0
    student_fam_ok = 0
    for pid in common:
        g_fam, g_sub = truth[pid]
        if frontier[pid]["family"] == g_fam and not frontier[pid]["abstain"]:
            frontier_fam_ok += 1
            if frontier[pid]["subtype"] == g_sub:
                frontier_sub_fam_ok += 1
        if student[pid]["family"] == g_fam and not student[pid]["abstain"]:
            student_fam_ok += 1
            if student[pid]["subtype"] == g_sub:
                student_sub_fam_ok += 1
    rep["subtype_given_family_correct"] = {
        "frontier": frontier_sub_fam_ok / max(1, frontier_fam_ok),
        "student":  student_sub_fam_ok / max(1, student_fam_ok),
        "frontier_n_fam_ok": frontier_fam_ok,
        "student_n_fam_ok": student_fam_ok,
    }

    op = Path(args.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(rep, indent=2) + "\n")

    print(json.dumps({k: v for k, v in rep.items() if k != "per_class"}, indent=2))
    print(f"\n[fc] wrote {op}")


if __name__ == "__main__":
    main()
