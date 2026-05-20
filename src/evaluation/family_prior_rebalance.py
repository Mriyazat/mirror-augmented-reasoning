"""CPU-only F1 booster: per-family rank re-balancing.

Idea: the student is heavily biased toward AdverseRisk (over-predicts by
2.5x on balanced test) because AdverseRisk is 43% of the training
distribution.  We do NOT have raw class logits (label_dist is {}), but
we have:
  - the predicted family
  - the confidence
  - the trace_majority
  - (across rerank decodes) the family vote distribution

Strategy: compute per-family bias correction factors on val5k by
inverting the over/under-prediction ratio. For each test pair, if the
predicted family is over-represented (e.g. AdverseRisk) AND the
trace_majority disagrees, switch to trace_majority. If trace_majority
also agrees with the over-represented family, defer to the second-best
voter (greedy mode prediction).

This is a per-family soft rebalance, not a hard cap. Tuned to maximize
macro-F1 on val5k, applied to test.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]


def _trace_majority(trace):
    if not isinstance(trace, dict):
        return None
    hints = [s.get("family_hint") for s in (trace.get("steps") or [])
             if s.get("family_hint") in FAMS]
    if not hints:
        return None
    c = Counter(hints).most_common(2)
    return c[0][0]


def _load_preds(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("input_order") or "ab") != "ab":
                continue
            pid = r["pair_id"]
            fp = r.get("final_prediction") or {}
            out[pid] = {
                "final": fp.get("family") if fp.get("family") in FAMS else None,
                "abstain": bool(fp.get("abstain", False)),
                "conf": float(fp.get("confidence") or 0.5),
                "trace_maj": _trace_majority(r.get("trace")),
            }
    return out


def _load_truth(manifest_path):
    rows = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet",
                         columns=["pair_id", "family"]).to_pylist()
    fmap = {r["pair_id"]: r["family"] for r in rows}
    if manifest_path is None:
        return fmap
    with open(manifest_path) as f:
        keep = set(json.loads(l)["pair_id"] for l in f)
    return {pid: fmap[pid] for pid in keep if pid in fmap}


def _macro(yt, yp):
    return f1_score(yt, yp, labels=FAMS, average="macro", zero_division=0)


def calibrate_bias(rerank_path, greedy_path, manifest_path):
    """Returns the per-family bias-correction factors fit on this dataset."""
    truth = _load_truth(manifest_path)
    rerank = _load_preds(rerank_path)
    greedy = _load_preds(greedy_path)
    common = sorted(set(rerank) & set(greedy) & set(truth))

    pred_counts = Counter()
    truth_counts = Counter()
    for pid in common:
        truth_counts[truth[pid]] += 1
        f = rerank[pid]["final"]
        if f in FAMS and not rerank[pid]["abstain"]:
            pred_counts[f] += 1

    # Bias factor = predicted / truth.  >1 = over-predicted, <1 = under.
    bias = {}
    for f in FAMS:
        bias[f] = pred_counts[f] / max(1, truth_counts[f])
    return bias, common, rerank, greedy, truth


def apply_anti_bias(rerank, greedy, truth, common, bias, over_threshold=1.3,
                    conf_max=1.0, require_agreement=True):
    """Smart rule: only switch when ALL of:
       1. predicted family is over-predicted (bias > over_threshold)
       2. rerank confidence < conf_max (we're not sure)
       3. trace_majority AND greedy_final agree on a different alternative
          (two independent voters agree it's NOT the over-predicted class)
    """
    yt = [truth[pid] for pid in common]
    yp_baseline = [rerank[pid]["final"] or "PD_Activity" for pid in common]

    yp_rebal = []
    n_switched = 0
    for pid in common:
        r = rerank[pid]
        g = greedy[pid]
        f = r["final"] or "PD_Activity"
        new_f = f

        if (f in FAMS
                and bias.get(f, 1.0) > over_threshold
                and r["conf"] < conf_max):
            tm = r["trace_maj"]
            gf = g["final"]
            if require_agreement:
                if tm in FAMS and gf in FAMS and tm == gf and tm != f:
                    new_f = tm
            else:
                if tm in FAMS and tm != f:
                    new_f = tm
                elif gf in FAMS and gf != f:
                    new_f = gf

        if new_f != f:
            n_switched += 1
        yp_rebal.append(new_f)
    return yt, yp_baseline, yp_rebal, n_switched


def main():
    # ---- Fit on val5k ----
    val_rerank = ROOT / "outputs/student/trace_align/rescue_data/preds_val5k_rerank4.jsonl"
    val_greedy = ROOT / "outputs/student/trace_align/rescue_data/preds_val5k_greedy.jsonl"
    bias, common, rerank, greedy, truth = calibrate_bias(val_rerank, val_greedy, None)
    print("Per-family bias on val5k (>1 = over-predict, <1 = under-predict):")
    for f, b in sorted(bias.items(), key=lambda x: -x[1]):
        print(f"  {f:18s} {b:.3f}")
    print()

    best_cfg = None
    best_f1 = -1
    print("=== mode A: REQUIRE 2-voter agreement (trace_maj == greedy_final) ===")
    for thr in [1.5, 2.0]:
        for conf_max in [0.75, 0.80, 0.85]:
            yt, yp_b, yp_r, n_sw = apply_anti_bias(
                rerank, greedy, truth, common, bias,
                over_threshold=thr, conf_max=conf_max, require_agreement=True,
            )
            f1_b = _macro(yt, yp_b)
            f1_r = _macro(yt, yp_r)
            print(f"  thr={thr:.1f} conf<{conf_max:.2f}  "
                  f"baseline={f1_b:.4f}  rebal={f1_r:.4f}  Δ={f1_r-f1_b:+.4f}  switched={n_sw}")
            if f1_r > best_f1:
                best_f1 = f1_r
                best_cfg = (thr, conf_max, True)

    print("\n=== mode B: trace_maj alone (no greedy needed; works on cold splits) ===")
    for thr in [1.5, 2.0]:
        for conf_max in [0.70, 0.75, 0.80, 0.85]:
            yt, yp_b, yp_r, n_sw = apply_anti_bias(
                rerank, greedy, truth, common, bias,
                over_threshold=thr, conf_max=conf_max, require_agreement=False,
            )
            f1_b = _macro(yt, yp_b)
            f1_r = _macro(yt, yp_r)
            print(f"  thr={thr:.1f} conf<{conf_max:.2f}  "
                  f"baseline={f1_b:.4f}  rebal={f1_r:.4f}  Δ={f1_r-f1_b:+.4f}  switched={n_sw}")
            if f1_r > best_f1:
                best_f1 = f1_r
                best_cfg = (thr, conf_max, False)

    print(f"\n[best] thr={best_cfg[0]:.1f} conf_max={best_cfg[1]:.2f} "
          f"require_agreement={best_cfg[2]} → val F1={best_f1:.4f}\n")

    # ---- Apply to test splits using the val-fit bias ----
    print()
    test_splits = [
        ("random_full",
         ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank8_abba.jsonl",
         ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank4_abba.jsonl",
         ROOT / "outputs/eval_prompts/random_full_test_5000_stratified.manifest.jsonl"),
        ("drug_cold",
         ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_drug_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
         ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_drug_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
         ROOT / "outputs/eval_prompts/drug_cold_test_5000_stratified.manifest.jsonl"),
        ("pair_cold",
         ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_pair_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
         ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_pair_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
         ROOT / "outputs/eval_prompts/pair_cold_test_5000_stratified.manifest.jsonl"),
    ]

    out_summary = {"val_bias": bias, "test_results": {}}
    thr, conf_max, require_agreement = best_cfg
    for split, primary, greedyish, mani in test_splits:
        t_truth = _load_truth(mani)
        t_primary = _load_preds(primary)
        t_greedy = _load_preds(greedyish)
        t_common = sorted(set(t_primary) & set(t_greedy) & set(t_truth))
        yt, yp_b, yp_r, n_sw = apply_anti_bias(
            t_primary, t_greedy, t_truth, t_common, bias,
            over_threshold=thr, conf_max=conf_max, require_agreement=require_agreement,
        )
        f1_b = _macro(yt, yp_b)
        f1_r = _macro(yt, yp_r)
        print(f"{split:14s}  n={len(t_common)}  "
              f"baseline F1={f1_b:.4f}  rebalanced F1={f1_r:.4f}  "
              f"Δ={f1_r-f1_b:+.4f}  switched={n_sw}")
        out_summary["test_results"][split] = {
            "baseline_F1": f1_b,
            "rebalanced_F1": f1_r,
            "delta": f1_r - f1_b,
            "n_switched": n_sw,
        }

    out = ROOT / "outputs/diag2/headline/family_rebalance.json"
    out.write_text(json.dumps(out_summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
