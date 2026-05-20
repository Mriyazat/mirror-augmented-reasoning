"""How much F1 lift is possible if trace-align SFT perfectly fixes
every 'trace_maj == gold AND final != gold' case?

Computes on val5k:
  - baseline rerank4 macro-F1
  - oracle-fix-all-rescues macro-F1 (theoretical ceiling)
  - oracle-fix 30% / 50% / 70% of rescues (realistic ranges)
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
                "trace_maj": _trace_majority(r.get("trace")),
            }
    return out


def main():
    labels = {r["pair_id"]: r["family"]
              for r in pq.read_table(
                  ROOT / "data_processed/labels_hierarchical.parquet",
                  columns=["pair_id", "family"]).to_pylist()}

    paths = {
        "rerank4": ROOT / "outputs/student/trace_align/rescue_data/preds_val5k_rerank4.jsonl",
        "greedy":  ROOT / "outputs/student/trace_align/rescue_data/preds_val5k_greedy.jsonl",
    }
    preds = {name: _load_preds(p) for name, p in paths.items()}
    common = sorted(set(preds["rerank4"]) & set(preds["greedy"]) & set(labels))

    print(f"n_common = {len(common)}\n")

    for name, p in preds.items():
        yt = [labels[pid] for pid in common]
        yp = [p[pid]["final"] or "PD_Activity" for pid in common]
        f1 = f1_score(yt, yp, labels=FAMS, average="macro", zero_division=0)
        rescue_ids = [pid for pid in common
                      if p[pid]["trace_maj"] == labels[pid]
                      and p[pid]["final"] != labels[pid]
                      and not p[pid]["abstain"]]
        print(f"--- {name} ---")
        print(f"  baseline F1               : {f1:.4f}")
        print(f"  n_rescue_candidates       : {len(rescue_ids)}  "
              f"({len(rescue_ids)/len(common):.1%} of test)")

        # Oracle: fix every rescue
        oracle_pred = list(yp)
        for j, pid in enumerate(common):
            if pid in set(rescue_ids):
                oracle_pred[j] = labels[pid]
        f1_oracle = f1_score(yt, oracle_pred, labels=FAMS, average="macro", zero_division=0)
        print(f"  ORACLE fix-all-rescues F1 : {f1_oracle:.4f}  (Δ = +{f1_oracle - f1:.4f})")

        # Realistic ranges
        for hit_rate in [0.30, 0.50, 0.70]:
            n_fix = int(len(rescue_ids) * hit_rate)
            pred = list(yp)
            rng = np.random.default_rng(0)
            chosen = set(rng.choice(rescue_ids, size=n_fix, replace=False))
            for j, pid in enumerate(common):
                if pid in chosen:
                    pred[j] = labels[pid]
            f1r = f1_score(yt, pred, labels=FAMS, average="macro", zero_division=0)
            print(f"  {int(hit_rate*100)}% SFT hit rate F1     : {f1r:.4f}  (Δ = +{f1r - f1:.4f})")
        print()


if __name__ == "__main__":
    main()
