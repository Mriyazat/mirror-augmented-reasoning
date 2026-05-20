"""Confusion-aware router that fixes the LLM's biggest pair_cold mistakes.

Validation-tuned policy:
  For each (gold_family g, predicted_family p) confusion bucket on val:
    - If, *conditioned* on LLM predicting p, switching to the baseline's
      prediction would improve macro-F1 on val, mark (p -> baseline) as a
      rewrite rule.
  Apply the rewrite rules on held-out test.

Hyperparameters tuned on val:
  - baseline choice (XGB / MLP)
  - LLM-confidence threshold to allow the rewrite
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]


def _manifest(path: Path) -> list[str]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)["pair_id"])
    return out


def _truth(path: Path, keep: set[str]) -> dict[str, str]:
    rows = pq.read_table(path, columns=["pair_id", "family"]).to_pylist()
    return {r["pair_id"]: r["family"] for r in rows if r["pair_id"] in keep}


def _preds(path: Path, keep: set[str]) -> dict[str, dict]:
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("pair_id")
            if pid not in keep:
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
                "abstain": bool(fp.get("abstain", False)),
                "conf": conf,
            }
    return out


def _macro(yt, yp, n):
    return float(f1_score(yt, yp, labels=list(range(n)), average="macro", zero_division=0))


def _arr(pids, truth, preds, lab2idx):
    sentinel = len(lab2idx)
    yt = np.asarray([lab2idx[truth[pid]] for pid in pids])
    yp = np.asarray(
        [
            sentinel if (preds[pid]["abstain"] or preds[pid]["family"] not in lab2idx)
            else lab2idx[preds[pid]["family"]]
            for pid in pids
        ]
    )
    return yt, yp


def _llm_conf(pids, preds):
    return np.asarray([preds[pid]["conf"] for pid in pids])


def _learn_rules(yv, yv_llm, yv_base, yv_conf, conf_threshold, n_classes):
    """
    For each LLM-predicted class p, decide whether routing to baseline
    *only when* LLM is below conf_threshold improves val macro-F1.
    Returns set of (p -> use_base) rules.
    """
    base_route = set()
    for p in range(n_classes):
        mask = (yv_llm == p) & (yv_conf < conf_threshold)
        if mask.sum() < 20:
            continue
        # Baseline F1 contribution if we route this bucket
        candidate = np.where(mask, yv_base, yv_llm)
        if _macro(yv, candidate, n_classes) > _macro(yv, yv_llm, n_classes):
            base_route.add(p)
    return base_route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_jsonl", required=True)
    ap.add_argument("--pred_llm", required=True)
    ap.add_argument("--pred_base", required=True)
    ap.add_argument("--base_name", default="BASE")
    ap.add_argument("--labels", default="data_processed/labels_hierarchical.parquet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_fraction", type=float, default=0.5)
    ap.add_argument(
        "--conf_grid",
        default="0.0,0.6,0.65,0.7,0.75,0.8,0.85,0.9,1.0",
        help="LLM confidence thresholds searched on val.",
    )
    ap.add_argument("--output", default=None)
    ap.add_argument("--output_predictions", default=None,
                    help="Optional JSONL path to dump router predictions (test split only).")
    args = ap.parse_args()

    pids_all = _manifest(Path(args.manifest_jsonl))
    keep = set(pids_all)
    truth = _truth(Path(args.labels), keep)
    llm = _preds(Path(args.pred_llm), keep)
    base = _preds(Path(args.pred_base), keep)
    common = sorted(set(truth) & set(llm) & set(base))
    if not common:
        raise SystemExit("[router] no overlap.")

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(common))
    rng.shuffle(idx)
    n_val = int(len(common) * args.val_fraction)
    val_ids = [common[i] for i in idx[:n_val]]
    tst_ids = [common[i] for i in idx[n_val:]]
    labels_list = sorted(set(truth.values()))
    lab2idx = {f: i for i, f in enumerate(labels_list)}
    n_classes = len(labels_list)

    yv, yv_llm = _arr(val_ids, truth, llm, lab2idx)
    _, yv_base = _arr(val_ids, truth, base, lab2idx)
    yt, yt_llm = _arr(tst_ids, truth, llm, lab2idx)
    _, yt_base = _arr(tst_ids, truth, base, lab2idx)
    cv = _llm_conf(val_ids, llm)
    ct = _llm_conf(tst_ids, llm)

    grid = [float(x) for x in args.conf_grid.split(",") if x.strip()]
    best = {"val": -1.0, "tau": None, "rules": set()}
    for tau in grid:
        rules = _learn_rules(yv, yv_llm, yv_base, cv, tau, n_classes)
        # apply on val
        apply_mask = np.isin(yv_llm, list(rules)) & (cv < tau)
        yp = np.where(apply_mask, yv_base, yv_llm)
        f = _macro(yv, yp, n_classes)
        if f > best["val"]:
            best = {"val": f, "tau": tau, "rules": rules}

    # Apply best policy to test
    apply_mask_t = np.isin(yt_llm, list(best["rules"])) & (ct < best["tau"])
    yp_router = np.where(apply_mask_t, yt_base, yt_llm)

    out = {
        "manifest": str(args.manifest_jsonl),
        "base_name": args.base_name,
        "n_val": len(val_ids),
        "n_test": len(tst_ids),
        "best_tau": best["tau"],
        "rewrite_rules": sorted([labels_list[r] for r in best["rules"]]),
        "val_macro_router": best["val"],
        "test_macro_llm_solo": _macro(yt, yt_llm, n_classes),
        "test_macro_base_solo": _macro(yt, yt_base, n_classes),
        "test_macro_router": _macro(yt, yp_router, n_classes),
        "test_n_rewritten": int(apply_mask_t.sum()),
        "test_n_total": int(len(tst_ids)),
        "seed": args.seed,
    }

    print(
        f"[crouter] tau*={best['tau']}  rules={out['rewrite_rules']}\n"
        f"  TEST: LLM={out['test_macro_llm_solo']:.4f}  "
        f"{args.base_name}={out['test_macro_base_solo']:.4f}  router={out['test_macro_router']:.4f}\n"
        f"  rewrites applied to {out['test_n_rewritten']} / {out['test_n_total']} test pairs"
    )

    if args.output:
        op = Path(args.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2) + "\n")
        print(f"[crouter] wrote {op}")

    if args.output_predictions:
        idx2lab = {i: l for l, i in lab2idx.items()}
        op = Path(args.output_predictions)
        op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w") as f:
            for pid, y in zip(tst_ids, yp_router):
                fam = idx2lab.get(int(y), None)
                rec = {
                    "pair_id": pid,
                    "input_order": "ab",
                    "final_prediction": {
                        "family": fam,
                        "abstain": fam is None,
                    },
                }
                f.write(json.dumps(rec) + "\n")
        print(f"[crouter] wrote test predictions {op}")


if __name__ == "__main__":
    main()
