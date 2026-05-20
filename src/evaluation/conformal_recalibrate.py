"""Per-family conformal recalibration on val1k -> evaluate on all test splits.

Fits per-family confidence thresholds that target selective-accuracy levels
{0.85, 0.90, 0.95} with a minimum coverage floor, then evaluates the
resulting selective accuracy / coverage / macro-F1 on each test split.

Outputs a JSON summary + a paper table.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]


def _truth():
    rows = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet",
                         columns=["pair_id", "family"]).to_pylist()
    return {r["pair_id"]: r["family"] for r in rows}


def _load_pred_ab(p):
    out = {}
    if not Path(p).exists():
        return out
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            pid = r["pair_id"]
            if (r.get("input_order") or "ab") != "ab":
                continue
            if pid in out:
                continue
            fp = r.get("final_prediction") or {}
            out[pid] = {
                "family": fp.get("family"),
                "conf": float(fp.get("confidence", 0.0)) if fp.get("confidence") is not None else 0.0,
                "abstain": bool(fp.get("abstain", False)),
            }
    return out


def _manifest(p):
    s = set()
    if not Path(p).exists():
        return s
    with open(p) as f:
        for line in f:
            s.add(json.loads(line)["pair_id"])
    return s


def fit_per_family(val_pred, truth, sel_acc_target, min_coverage):
    """For each family, find smallest tau such that sel-acc >= target.

    If unreachable, fall back to tau that yields min_coverage.
    """
    by_fam = defaultdict(list)
    for pid, pr in val_pred.items():
        if pid not in truth or pr["abstain"] or pr["family"] not in FAMS:
            continue
        by_fam[pr["family"]].append((pr["conf"], 1 if pr["family"] == truth[pid] else 0))

    thresholds = {}
    for fam in FAMS:
        items = sorted(by_fam.get(fam, []), reverse=True)
        if not items:
            thresholds[fam] = 1.01
            continue
        confs = np.array([c for c, _ in items])
        correct = np.array([y for _, y in items], dtype=float)
        cum_correct = np.cumsum(correct)
        cum_n = np.arange(1, len(items) + 1)
        sel_acc = cum_correct / cum_n
        coverage = cum_n / len(items)
        cutoff_idx = None
        for i in range(len(items) - 1, -1, -1):
            if sel_acc[i] >= sel_acc_target and coverage[i] >= min_coverage:
                cutoff_idx = i
                break
        if cutoff_idx is None:
            tgt_n = int(np.ceil(min_coverage * len(items)))
            cutoff_idx = max(tgt_n - 1, 0)
        thresholds[fam] = float(confs[cutoff_idx])
    return thresholds


def apply_and_score(pred, truth, keep, thresholds):
    yt, yp, n_kept, n_total = [], [], 0, 0
    for pid in pred:
        if pid not in truth or pid not in keep:
            continue
        n_total += 1
        pr = pred[pid]
        if pr["abstain"] or pr["family"] not in FAMS:
            continue
        tau = thresholds.get(pr["family"], 1.01)
        if pr["conf"] < tau:
            continue
        yt.append(truth[pid])
        yp.append(pr["family"])
        n_kept += 1
    if not yt:
        return {"coverage": 0.0, "sel_acc": 0.0, "macro_F1": 0.0, "n_kept": 0, "n_total": n_total}
    acc = sum(1 for a, b in zip(yt, yp) if a == b) / len(yt)
    from sklearn.metrics import f1_score
    macro = f1_score(yt, yp, labels=FAMS, average="macro", zero_division=0)
    return {
        "coverage": n_kept / n_total,
        "sel_acc": acc,
        "macro_F1": macro,
        "n_kept": n_kept,
        "n_total": n_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_pred", default=str(ROOT / "outputs/eval_prompts/pred_probe_val1k_rerank4.jsonl"))
    ap.add_argument("--targets", nargs="+", type=float, default=[0.85, 0.90, 0.95])
    ap.add_argument("--min_coverage", type=float, default=0.25)
    ap.add_argument("--output_dir", default=str(ROOT / "outputs/diag2/conformal_recal"))
    args = ap.parse_args()

    truth = _truth()
    val_pred = _load_pred_ab(args.val_pred)
    print(f"[conf] val records loaded: {len(val_pred)}")

    test_sources = {
        "random_full": ROOT / "outputs/eval_prompts/pred_cpu_stack_random_full.jsonl",
        "drug_cold":   ROOT / "outputs/eval_prompts/pred_cpu_stack_drug_cold.jsonl",
        "pair_cold":   ROOT / "outputs/eval_prompts/pred_cpu_stack_pair_cold.jsonl",
    }
    manifests = {sp: ROOT / f"outputs/eval_prompts/{sp}_test_5000_stratified.manifest.jsonl" for sp in test_sources}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    print(f"\n{'target':>10s} {'split':>14s} {'coverage':>10s} {'sel_acc':>10s} {'macroF1':>10s}")
    print("-" * 60)
    for target in args.targets:
        thr = fit_per_family(val_pred, truth, target, args.min_coverage)
        rows = {"thresholds": thr, "splits": {}}
        for sp, src in test_sources.items():
            pred = _load_pred_ab(src)
            keep = _manifest(manifests[sp])
            res = apply_and_score(pred, truth, keep, thr)
            rows["splits"][sp] = res
            print(f"{target:>10.2f} {sp:>14s} {res['coverage']:>10.3f} {res['sel_acc']:>10.3f} {res['macro_F1']:>10.3f}")
        summary[f"target_{target:.2f}"] = rows
        (out_dir / f"target_{int(target*100)}.json").write_text(json.dumps(rows, indent=2) + "\n")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
