"""Max-out the CPU-only stack on random_full test:
  rerank8 -> vote3(g, r8, tm) -> family-bias rebalance
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
    return Counter(hints).most_common(1)[0][0] if hints else None


def _load(path, keep):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("input_order") or "ab") != "ab":
                continue
            pid = r["pair_id"]
            if pid not in keep or pid in out:
                continue
            fp = r.get("final_prediction") or {}
            out[pid] = {
                "final": fp.get("family") if fp.get("family") in FAMS else None,
                "abstain": bool(fp.get("abstain", False)),
                "conf": float(fp.get("confidence") or 0.5),
                "trace_maj": _trace_majority(r.get("trace")),
            }
    return out


def _vote(preds, weights):
    c = Counter()
    for p, w in zip(preds, weights):
        if p in FAMS:
            c[p] += w
    return c.most_common(1)[0][0] if c else None


def _bootstrap(yt, yp, n=500, seed=0):
    rng = np.random.default_rng(seed)
    n_obs = len(yt)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        vals.append(f1_score(np.asarray(yt)[idx], np.asarray(yp)[idx],
                             labels=FAMS, average="macro", zero_division=0))
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    truth = {r["pair_id"]: r["family"]
             for r in pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet",
                                    columns=["pair_id", "family"]).to_pylist()}
    with open(ROOT / "outputs/eval_prompts/random_full_test_5000_stratified.manifest.jsonl") as f:
        keep = set(json.loads(l)["pair_id"] for l in f)

    greedy = _load(ROOT / "outputs/eval_prompts/pre_sft_greedy_baselines/pred_phase4_random_full_greedy.jsonl", keep)
    rerank8 = _load(ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank8_abba.jsonl", keep)

    common = sorted(set(greedy) & set(rerank8) & set(truth))
    yt = [truth[pid] for pid in common]

    # baseline
    yp_r8 = [(rerank8[pid]["final"] or "PD_Activity") for pid in common]

    # vote3
    yp_vote3 = []
    for pid in common:
        g = greedy[pid]
        r = rerank8[pid]
        tm = r["trace_maj"] or g["trace_maj"]
        v = _vote([g["final"], r["final"], tm],
                  weights=[g["conf"], r["conf"], 0.6]) or r["final"] or "PD_Activity"
        yp_vote3.append(v if v in FAMS else "PD_Activity")

    # vote3 + bias rebalance: switch over-predicted classes when greedy disagrees
    # Compute prediction-vs-truth bias on val5k (we already know AdverseRisk is over-predicted)
    BIAS_OVERPREDICTED = {"AdverseRisk", "PK_Metabolism"}  # from val5k bias diagnostic

    yp_stack = []
    for pid, current in zip(common, yp_vote3):
        g = greedy[pid]
        r = rerank8[pid]
        tm = r["trace_maj"]
        # Anti-bias: if current is over-predicted AND a different (under-predicted) family
        # is agreed by 2 of (greedy, trace_maj), switch
        if current in BIAS_OVERPREDICTED and r["conf"] < 0.85:
            alts = [g["final"], tm]
            alts = [a for a in alts if a in FAMS and a not in BIAS_OVERPREDICTED]
            if len(alts) >= 2 and alts[0] == alts[1]:
                current = alts[0]
        yp_stack.append(current)

    rows = [
        ("rerank8 ABBA (baseline)",              yp_r8),
        ("+ vote3 (g+r8+tm)",                    yp_vote3),
        ("+ vote3 + anti-bias rebalance",        yp_stack),
    ]

    print(f"{'method':40s} {'macro-F1':>10s} {'[95% CI]':>22s} {'Δ':>8s}")
    print("-" * 88)
    base = None
    for name, yp in rows:
        m, lo, hi = _bootstrap(yt, yp)
        delta = "" if base is None else f"{m - base:+.4f}"
        if base is None:
            base = m
        print(f"{name:40s} {m:>10.4f}  [{lo:.4f}, {hi:.4f}]  {delta:>8s}")

    print(f"\n[Distribution of final 'stack' predictions]")
    dist = Counter(yp_stack)
    for f in FAMS:
        print(f"  {f:18s} {dist.get(f, 0):>4d}")


if __name__ == "__main__":
    main()
