"""Honest val->test evaluation for hybrid routers.

Splits a manifest into val/test (deterministic shuffle), tunes router params on val,
reports final metrics on held-out test.
"""
from __future__ import annotations

import argparse
import json
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


def _load_preds(path: Path, keep: set[str]) -> dict[str, dict]:
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
            try:
                conf = float(fp.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            out[pid] = {
                "family": fp.get("family"),
                "abstain": bool(fp.get("abstain", False)),
                "conf": conf,
            }
    return out


def _macro(yt: np.ndarray, yp: np.ndarray, n_classes: int):
    return float(
        f1_score(yt, yp, labels=list(range(n_classes)), average="macro", zero_division=0)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_jsonl", required=True)
    ap.add_argument("--pred_llm", required=True)
    ap.add_argument("--pred_base", required=True)
    ap.add_argument("--llm_name", default="LLM")
    ap.add_argument("--base_name", default="BASE")
    ap.add_argument("--labels", default="data_processed/labels_hierarchical.parquet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_fraction", type=float, default=0.5)
    ap.add_argument("--tau_grid", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    pids = _manifest(Path(args.manifest_jsonl))
    keep = set(pids)
    truth = _truth(Path(args.labels), keep)
    llm = _load_preds(Path(args.pred_llm), keep)
    base = _load_preds(Path(args.pred_base), keep)
    common = sorted(set(truth) & set(llm) & set(base))
    if not common:
        raise SystemExit("[router] no overlap across truth+llm+base.")

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(common))
    rng.shuffle(idx)
    n_val = int(len(common) * args.val_fraction)
    val_ids = [common[i] for i in idx[:n_val]]
    tst_ids = [common[i] for i in idx[n_val:]]

    labels = sorted(set(truth.values()))
    lab2idx = {f: i for i, f in enumerate(labels)}
    sentinel = len(labels)

    def arr(pid_list):
        yt = np.asarray([lab2idx[truth[pid]] for pid in pid_list])
        yp_llm = np.asarray(
            [sentinel if (llm[pid]["abstain"] or llm[pid]["family"] not in lab2idx) else lab2idx[llm[pid]["family"]] for pid in pid_list]
        )
        yp_base = np.asarray(
            [sentinel if (base[pid]["abstain"] or base[pid]["family"] not in lab2idx) else lab2idx[base[pid]["family"]] for pid in pid_list]
        )
        c_base = np.asarray([base[pid]["conf"] for pid in pid_list])
        return yt, yp_llm, yp_base, c_base

    yv, llv, bav, cv = arr(val_ids)
    yt, llt, bat, ct = arr(tst_ids)
    n_classes = len(labels)

    taus = [float(x.strip()) for x in args.tau_grid.split(",") if x.strip()]
    best_tau = None
    best_val = -1.0
    for tau in taus:
        yp = np.where(cv >= tau, bav, llv)
        f = _macro(yv, yp, n_classes)
        if f > best_val:
            best_val = f
            best_tau = tau

    yp_test = np.where(ct >= best_tau, bat, llt)
    out = {
        "n_common": len(common),
        "n_val": len(val_ids),
        "n_test": len(tst_ids),
        "llm_name": args.llm_name,
        "base_name": args.base_name,
        "best_tau_val": best_tau,
        "val_macro_f1_router": best_val,
        "test_macro_f1": {
            args.llm_name: _macro(yt, llt, n_classes),
            args.base_name: _macro(yt, bat, n_classes),
            "router": _macro(yt, yp_test, n_classes),
        },
        "meta": {"seed": args.seed, "val_fraction": args.val_fraction},
    }

    print(
        f"[router] common={len(common):,} val={len(val_ids):,} test={len(tst_ids):,}\n"
        f"  tau*={best_tau:.2f}  val_router={best_val:.4f}\n"
        f"  test: {args.llm_name}={out['test_macro_f1'][args.llm_name]:.4f}  "
        f"{args.base_name}={out['test_macro_f1'][args.base_name]:.4f}  "
        f"router={out['test_macro_f1']['router']:.4f}"
    )

    if args.output:
        op = Path(args.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2) + "\n")
        print(f"[router] wrote {op}")


if __name__ == "__main__":
    main()
