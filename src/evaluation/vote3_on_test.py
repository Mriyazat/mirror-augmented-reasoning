"""Apply the vote-3 ensemble (greedy + rerank8 + trace_maj) to the
random_full TEST split now that we have a true greedy file for the
pre-SFT student.
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
    return Counter(hints).most_common(1)[0][0]


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
    if not c:
        return None
    return c.most_common(1)[0][0]


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
    rerank4 = _load(ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank4_abba.jsonl", keep)

    common = sorted(set(greedy) & set(rerank8) & set(rerank4) & set(truth))
    print(f"n_common = {len(common)}\n")

    yt = [truth[pid] for pid in common]
    yp_greedy  = [(greedy[pid]["final"]  or "PD_Activity") for pid in common]
    yp_rerank4 = [(rerank4[pid]["final"] or "PD_Activity") for pid in common]
    yp_rerank8 = [(rerank8[pid]["final"] or "PD_Activity") for pid in common]

    # vote-3: greedy + rerank8 + trace_maj (using rerank8's trace)
    yp_vote3 = []
    for pid in common:
        g = greedy[pid]
        r = rerank8[pid]
        tm = r["trace_maj"] or g["trace_maj"]
        v = _vote([g["final"], r["final"], tm],
                  weights=[g["conf"], r["conf"], 0.6]) or r["final"] or "PD_Activity"
        yp_vote3.append(v if v in FAMS else "PD_Activity")

    # vote-3+: also include rerank4 (4-voter)
    yp_vote4 = []
    for pid in common:
        g = greedy[pid]
        r4 = rerank4[pid]
        r8 = rerank8[pid]
        tm = r8["trace_maj"] or g["trace_maj"]
        v = _vote([g["final"], r4["final"], r8["final"], tm],
                  weights=[g["conf"], r4["conf"], r8["conf"], 0.6]) or r8["final"] or "PD_Activity"
        yp_vote4.append(v if v in FAMS else "PD_Activity")

    rows = [
        ("greedy alone",            yp_greedy),
        ("rerank4 ABBA",            yp_rerank4),
        ("rerank8 ABBA",            yp_rerank8),
        ("vote3 (g+r8+tm)",         yp_vote3),
        ("vote4 (g+r4+r8+tm)",      yp_vote4),
    ]

    print(f"{'method':28s} {'macro-F1':>10s} {'[95% CI]':>22s}")
    print("-" * 65)
    for name, yp in rows:
        m, lo, hi = _bootstrap(yt, yp)
        print(f"{name:28s} {m:>10.4f}  [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
