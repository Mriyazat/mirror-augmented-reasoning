"""Grid-search trace-rescue parameters on a 50% val slice; report best on 50% test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score

from src.inference.trace_rescue import rescue, FAMS


ROOT = Path(__file__).resolve().parents[2]


def _truth():
    rows = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet",
                         columns=["pair_id", "family"]).to_pylist()
    return {r["pair_id"]: r["family"] for r in rows}


def _load(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("pair_id")
            if (r.get("input_order") or "ab") != "ab":
                continue
            if pid in out:
                continue
            out[pid] = r
    return out


def _manifest(p):
    s = []
    with open(p) as f:
        for line in f:
            s.append(json.loads(line)["pair_id"])
    return s


def _apply(records, policy, min_steps, min_strength, max_conf, max_original_frac):
    """Returns {pid -> rescued_family_or_original}."""
    out = {}
    n_rescued = 0
    for pid, r in records.items():
        fa = r.get("final_prediction") or {}
        orig = fa.get("family")
        rule, new = rescue(r, policy, min_steps, min_strength, max_conf, max_original_frac)
        out[pid] = new if new else orig
        if new:
            n_rescued += 1
    return out, n_rescued


def macro(pred_map, truth, pids):
    f2i = {f: i for i, f in enumerate(FAMS)}
    sentinel = len(FAMS)
    yt = np.asarray([f2i[truth[p]] for p in pids])
    yp = np.asarray([
        sentinel if (pred_map[p] not in f2i) else f2i[pred_map[p]]
        for p in pids
    ])
    return f1_score(yt, yp, labels=list(range(len(FAMS))), average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_fraction", type=float, default=0.5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    truth = _truth()
    records = _load(args.predictions)
    keep = set(_manifest(args.manifest))
    common = sorted({pid for pid in records if pid in truth and pid in keep})
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(common)); rng.shuffle(idx)
    n_val = int(len(common) * args.val_fraction)
    val_ids = [common[i] for i in idx[:n_val]]
    tst_ids = [common[i] for i in idx[n_val:]]

    # Baseline (no rescue) on test
    baseline_pred = {pid: (records[pid].get("final_prediction") or {}).get("family") for pid in common}
    base_val = macro(baseline_pred, truth, val_ids)
    base_tst = macro(baseline_pred, truth, tst_ids)

    grid = []
    for policy in ["hint_majority", "conclusion_text", "hybrid"]:
        for max_conf in [0.4, 0.5, 0.6, 0.7, 1.0]:
            for min_strength in [0.5, 0.6, 0.7, 0.8, 1.0]:
                for max_orig in [0.10, 0.20, 0.34, 1.0]:
                    for min_steps in [2, 3, 4]:
                        grid.append((policy, max_conf, min_strength, max_orig, min_steps))

    best = None
    for params in grid:
        policy, max_conf, min_strength, max_orig, min_steps = params
        pmap, n_resc = _apply(
            {pid: records[pid] for pid in val_ids},
            policy, min_steps, min_strength, max_conf, max_orig,
        )
        full_pmap = {**baseline_pred, **pmap}
        f_val = macro(full_pmap, truth, val_ids)
        gain_val = f_val - base_val
        if best is None or gain_val > best["gain_val"]:
            best = {
                "params": params, "gain_val": float(gain_val),
                "f_val": float(f_val), "n_rescued_val": int(n_resc),
            }

    policy, max_conf, min_strength, max_orig, min_steps = best["params"]
    pmap_test, n_resc_test = _apply(
        {pid: records[pid] for pid in tst_ids},
        policy, min_steps, min_strength, max_conf, max_orig,
    )
    full_test = {**baseline_pred, **pmap_test}
    f_test = macro(full_test, truth, tst_ids)

    out = {
        "predictions": args.predictions,
        "n_val": len(val_ids), "n_test": len(tst_ids),
        "baseline_macroF1_val": base_val,
        "baseline_macroF1_test": base_tst,
        "best_params": {
            "policy": policy, "max_conf": max_conf,
            "min_strength": min_strength, "max_original_frac": max_orig,
            "min_steps": min_steps,
        },
        "best_val_macroF1": best["f_val"],
        "best_val_gain": best["gain_val"],
        "best_val_rescued": best["n_rescued_val"],
        "test_macroF1_rescued": float(f_test),
        "test_macroF1_delta": float(f_test - base_tst),
        "test_n_rescued": int(n_resc_test),
    }
    print(json.dumps(out, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
