"""2-decoder vote on val5k: greedy + rerank4 + (optional) trace-majority.

Reports macro-F1 and accuracy for:
  - greedy alone
  - rerank4 alone
  - vote(greedy, rerank4)
  - vote(greedy, rerank4, trace-majority)
  - trace-coherent-rescue: if trace_maj == rerank4_final, keep rerank4;
    else if trace_maj agrees with greedy, use greedy.

Output:
  outputs/student/trace_align/rescue_data/vote_val5k.json   (numbers)
  printed table to stdout
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score, accuracy_score

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
                "final": fp.get("family"),
                "abstain": bool(fp.get("abstain", False)),
                "trace_maj": _trace_majority(r.get("trace")),
                "conf": float(fp.get("confidence") or 0.5),
            }
    return out


def _vote(preds, weights=None):
    if weights is None:
        weights = [1.0] * len(preds)
    c = Counter()
    for p, w in zip(preds, weights):
        if p in FAMS:
            c[p] += w
    if not c:
        return None
    return c.most_common(1)[0][0]


def _bootstrap_ci(y_true, y_pred, n=500, seed=0):
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        vals.append(f1_score(np.asarray(y_true)[idx],
                             np.asarray(y_pred)[idx],
                             labels=FAMS, average="macro", zero_division=0))
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    rerank4 = _load_preds(
        ROOT / "outputs/student/trace_align/rescue_data/preds_val5k_rerank4.jsonl")
    greedy = _load_preds(
        ROOT / "outputs/student/trace_align/rescue_data/preds_val5k_greedy.jsonl")
    labels = {r["pair_id"]: r["family"]
              for r in pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet").to_pylist()}

    common = sorted(set(rerank4) & set(greedy) & set(labels))
    y_true = [labels[pid] for pid in common]

    def _ok(f):
        return f if f in FAMS else "PD_Activity"  # majority-class fallback for unparseable

    pred_greedy = [_ok(greedy[pid]["final"]) for pid in common]
    pred_rerank = [_ok(rerank4[pid]["final"]) for pid in common]

    pred_vote2 = []
    pred_vote3 = []
    pred_coherent = []
    for pid in common:
        g = greedy[pid]
        r = rerank4[pid]
        tm_g = g["trace_maj"]
        tm_r = r["trace_maj"]
        v2 = _vote([g["final"], r["final"]],
                   weights=[g["conf"], r["conf"]]) or r["final"] or "PD_Activity"
        pred_vote2.append(_ok(v2))
        v3 = _vote([g["final"], r["final"], tm_r or tm_g],
                   weights=[g["conf"], r["conf"], 0.6]) or r["final"] or "PD_Activity"
        pred_vote3.append(_ok(v3))
        if tm_r and tm_r == r["final"]:
            c = r["final"]
        elif tm_g and tm_g == g["final"]:
            c = g["final"]
        elif tm_r:
            c = tm_r
        else:
            c = r["final"] or "PD_Activity"
        pred_coherent.append(_ok(c))

    def report(name, yp):
        f1m, lo, hi = _bootstrap_ci(y_true, yp)
        acc = accuracy_score(y_true, yp)
        return (name, f1m, lo, hi, acc)

    rows = [
        report("greedy",                              pred_greedy),
        report("rerank4",                             pred_rerank),
        report("vote(greedy,rerank4)",                pred_vote2),
        report("vote(greedy,rerank4,trace_maj)",      pred_vote3),
        report("trace-coherent rescuer",              pred_coherent),
    ]

    print(f"{'method':36s}  macro-F1  [95% CI]              acc")
    print("-" * 78)
    for n, f1m, lo, hi, acc in rows:
        print(f"{n:36s}  {f1m:0.4f}    [{lo:0.4f}, {hi:0.4f}]   {acc:0.4f}")

    out = ROOT / "outputs/student/trace_align/rescue_data/vote_val5k.json"
    out.write_text(json.dumps([
        {"method": n, "macro_f1": f1m,
         "ci_lo": lo, "ci_hi": hi, "acc": acc}
        for (n, f1m, lo, hi, acc) in rows
    ], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
