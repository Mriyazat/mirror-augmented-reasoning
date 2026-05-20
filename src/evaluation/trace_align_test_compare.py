"""Compare pre-SFT vs post-SFT-v1 vs post-SFT-v2 on the random_full TEST split.

Tells us:
  1. Did trace-align SFT v1 actually move family F1 on test?
  2. Did v2-merged (rerank4 + greedy rescues) beat v1 (rerank4 only)?
  3. How did the family-prediction bias change after SFT?
  4. What's the per-family F1 delta?

Plus bootstrap CIs, so we know if any delta is statistically significant.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score, classification_report

ROOT = Path(__file__).resolve().parents[2]
FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]


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
            }
    return out


def _truth():
    rows = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet",
                         columns=["pair_id", "family"]).to_pylist()
    return {r["pair_id"]: r["family"] for r in rows}


def _macro(yt, yp):
    return f1_score(yt, yp, labels=FAMS, average="macro", zero_division=0)


def _bootstrap_ci(yt, yp, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(yt)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(f1_score(np.asarray(yt)[idx], np.asarray(yp)[idx],
                             labels=FAMS, average="macro", zero_division=0))
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _paired_bootstrap_diff(yt, yp_a, yp_b, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(yt)
    diffs = []
    yt_arr = np.asarray(yt)
    yp_a = np.asarray(yp_a)
    yp_b = np.asarray(yp_b)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        fa = f1_score(yt_arr[idx], yp_a[idx], labels=FAMS, average="macro", zero_division=0)
        fb = f1_score(yt_arr[idx], yp_b[idx], labels=FAMS, average="macro", zero_division=0)
        diffs.append(fb - fa)
    d = np.array(diffs)
    # 1-sided: prob that B <= A
    p = (d <= 0).mean()
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(p)


def main():
    truth = _truth()
    with open(ROOT / "outputs/eval_prompts/random_full_test_5000_stratified.manifest.jsonl") as f:
        keep = set(json.loads(l)["pair_id"] for l in f)

    files = {
        "pre-SFT (greedy)":  ROOT / "outputs/eval_prompts/pre_sft_greedy_baselines/pred_phase4_random_full_greedy.jsonl",
        "post-SFT-v1":       ROOT / "outputs/student/trace_align/eval_after/pred_traceAlign_random_full_greedy.jsonl",
        "post-SFT-v2-merged":ROOT / "outputs/student/trace_align/eval_after_v2/pred_traceAlign_v2_random_full_greedy.jsonl",
    }
    sources = {n: _load(p, keep) for n, p in files.items()}
    common = sorted(set.intersection(*[set(s.keys()) for s in sources.values()]) & set(truth.keys()))
    print(f"n_common = {len(common)}\n")

    results = {}
    for name, src in sources.items():
        yp = [(src[pid]["final"] or "PD_Activity") for pid in common]
        yt = [truth[pid] for pid in common]
        f1m, lo, hi = _bootstrap_ci(yt, yp)
        n_abstain = sum(1 for pid in common if src[pid]["abstain"])
        fam_dist = Counter(yp)
        results[name] = {"yt": yt, "yp": yp, "f1": f1m, "ci": [lo, hi],
                         "abstain": n_abstain, "dist": fam_dist}

    print(f"{'model':22s} {'macro-F1':>10s} {'[95% CI]':>22s} {'n_abstain':>10s}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:22s} {r['f1']:>10.4f}  [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]   {r['abstain']:>10d}")

    print("\n=== Paired bootstrap deltas (post-SFT minus pre-SFT) ===")
    base = results["pre-SFT (greedy)"]
    for name in ["post-SFT-v1", "post-SFT-v2-merged"]:
        d, dlo, dhi, p_le0 = _paired_bootstrap_diff(base["yt"], base["yp"], results[name]["yp"])
        sig = "***" if p_le0 < 0.01 else ("**" if p_le0 < 0.05 else ("*" if p_le0 < 0.10 else ""))
        print(f"  {name:22s} Δ={d:+.4f}  [{dlo:+.4f}, {dhi:+.4f}]  p(Δ<=0)={p_le0:.4f}  {sig}")

    print("\n=== Family-prediction distribution (balanced truth = ~715/family) ===")
    print(f"{'family':18s}  {'truth':>6s}  {'pre-SFT':>8s}  {'v1':>8s}  {'v2-merged':>10s}")
    truth_dist = Counter(results["pre-SFT (greedy)"]["yt"])
    for f in FAMS:
        print(f"  {f:18s} {truth_dist.get(f,0):>6d}  "
              f"{results['pre-SFT (greedy)']['dist'].get(f,0):>8d}  "
              f"{results['post-SFT-v1']['dist'].get(f,0):>8d}  "
              f"{results['post-SFT-v2-merged']['dist'].get(f,0):>10d}")

    print("\n=== Per-family F1 (pre-SFT vs post-SFT-v1 vs post-SFT-v2) ===")
    print(f"{'family':18s}  {'pre':>8s}  {'v1':>8s}  {'v2':>8s}")
    for f in FAMS:
        f1_pre = f1_score(results['pre-SFT (greedy)']['yt'], results['pre-SFT (greedy)']['yp'],
                          labels=[f], average="macro", zero_division=0)
        f1_v1 = f1_score(results['post-SFT-v1']['yt'], results['post-SFT-v1']['yp'],
                         labels=[f], average="macro", zero_division=0)
        f1_v2 = f1_score(results['post-SFT-v2-merged']['yt'], results['post-SFT-v2-merged']['yp'],
                         labels=[f], average="macro", zero_division=0)
        print(f"  {f:18s} {f1_pre:>8.4f}  {f1_v1:>8.4f}  {f1_v2:>8.4f}")

    # save summary
    out = {n: {"f1": r["f1"], "ci": r["ci"], "abstain": r["abstain"],
               "distribution": dict(r["dist"])} for n, r in results.items()}
    p = ROOT / "outputs/diag2/headline/trace_align_test_random_full.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
