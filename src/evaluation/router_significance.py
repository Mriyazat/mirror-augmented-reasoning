"""Paired bootstrap of router vs LLM-solo on the *same* held-out router-test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]


def _load_pred(path: Path) -> dict[str, str]:
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("pair_id")
            if (r.get("input_order") or "ab").lower() != "ab":
                continue
            if pid in out:
                continue
            fp = r.get("final_prediction") or {}
            out[pid] = fp.get("family")
    return out


def _truth(path: Path, keep: set[str]) -> dict[str, str]:
    rows = pq.read_table(path, columns=["pair_id", "family"]).to_pylist()
    return {r["pair_id"]: r["family"] for r in rows if r["pair_id"] in keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--labels", default="data_processed/labels_hierarchical.parquet")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    llm = _load_pred(Path(args.llm))
    rt = _load_pred(Path(args.router))
    common = sorted(set(llm) & set(rt))
    truth = _truth(Path(args.labels), set(common))
    common = [pid for pid in common if pid in truth]
    labels = sorted(set(truth.values()))
    lab2idx = {f: i for i, f in enumerate(labels)}
    sentinel = len(lab2idx)

    yt = np.asarray([lab2idx[truth[pid]] for pid in common])
    ya = np.asarray([lab2idx.get(llm[pid], sentinel) for pid in common])
    yb = np.asarray([lab2idx.get(rt[pid], sentinel) for pid in common])

    def macro(true_arr, pred_arr):
        return f1_score(true_arr, pred_arr, labels=list(range(len(labels))), average="macro", zero_division=0)

    base_a, base_b = macro(yt, ya), macro(yt, yb)
    rng = np.random.default_rng(args.seed)
    diffs = np.zeros(args.n_boot)
    n = len(common)
    for b in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = macro(yt[idx], yb[idx]) - macro(yt[idx], ya[idx])
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    p = float(min((diffs <= 0).mean(), (diffs >= 0).mean()) * 2)

    out = {
        "n_test": n,
        "macro_llm": base_a,
        "macro_router": base_b,
        "delta": base_b - base_a,
        "delta_ci95": [lo, hi],
        "p_value": p,
    }
    print(json.dumps(out, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
