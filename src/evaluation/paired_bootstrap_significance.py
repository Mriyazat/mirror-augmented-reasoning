"""Paired bootstrap significance test between two prediction files.

Reports whether model A significantly outperforms model B on:
  - macro-F1
  - rare-F1 (default rare families: PK_Absorption, PK_Distribution, PK_Excretion)

By default, abstentions are scored as wrong (sentinel class) for compulsory
comparison. Use --drop_abstain for selective-mode comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]


def _load_manifest(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    out: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.add(json.loads(line)["pair_id"])
    return out


def _load_truth(path: Path, keep: set[str] | None) -> dict[str, str]:
    rows = pq.read_table(path, columns=["pair_id", "family"]).to_pylist()
    out = {r["pair_id"]: r["family"] for r in rows}
    if keep is not None:
        out = {k: v for k, v in out.items() if k in keep}
    return out


def _load_preds(path: Path, keep: set[str] | None, use_ab_only: bool) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("pair_id")
            if not pid:
                continue
            if keep is not None and pid not in keep:
                continue
            order = (rec.get("input_order") or "ab").lower()
            if use_ab_only and order != "ab":
                continue
            if use_ab_only and pid in out:
                continue
            fp = rec.get("final_prediction") or {}
            out[pid] = {
                "family": fp.get("family"),
                "abstain": bool(fp.get("abstain", False)),
            }
    return out


def _build_arrays(
    truth: dict[str, str],
    pa: dict[str, dict],
    pb: dict[str, dict],
    drop_abstain: bool,
):
    labels = sorted(set(truth.values()))
    lab2idx = {f: i for i, f in enumerate(labels)}
    sentinel = len(labels)
    common = sorted(set(truth) & set(pa) & set(pb))
    yt = []
    ya = []
    yb = []
    for pid in common:
        gold = lab2idx[truth[pid]]
        ra = pa[pid]
        rb = pb[pid]
        if drop_abstain and (ra["abstain"] or rb["abstain"]):
            continue
        pa_idx = sentinel if (ra["abstain"] or ra["family"] not in lab2idx) else lab2idx[ra["family"]]
        pb_idx = sentinel if (rb["abstain"] or rb["family"] not in lab2idx) else lab2idx[rb["family"]]
        yt.append(gold)
        ya.append(pa_idx)
        yb.append(pb_idx)
    return np.asarray(yt), np.asarray(ya), np.asarray(yb), labels


def _scores(yt: np.ndarray, yp: np.ndarray, labels: list[str], rare_fams: list[str]):
    li = list(range(len(labels)))
    per = f1_score(yt, yp, labels=li, average=None, zero_division=0)
    macro = float(np.mean(per))
    idx = {f: i for i, f in enumerate(labels)}
    rare_idx = [idx[f] for f in rare_fams if f in idx]
    rare = float(np.mean([per[i] for i in rare_idx])) if rare_idx else 0.0
    return macro, rare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_a", required=True, help="Prediction file A.")
    ap.add_argument("--pred_b", required=True, help="Prediction file B.")
    ap.add_argument("--name_a", required=True)
    ap.add_argument("--name_b", required=True)
    ap.add_argument("--labels", default="data_processed/labels_hierarchical.parquet")
    ap.add_argument("--manifest_jsonl", default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--drop_abstain", action="store_true")
    ap.add_argument("--use_ab_only", action="store_true")
    ap.add_argument(
        "--rare_families",
        default="PK_Absorption,PK_Distribution,PK_Excretion",
        help="Comma-separated family names used for rare-F1.",
    )
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    keep = _load_manifest(Path(args.manifest_jsonl) if args.manifest_jsonl else None)
    truth = _load_truth(Path(args.labels), keep)
    pa = _load_preds(Path(args.pred_a), keep, args.use_ab_only)
    pb = _load_preds(Path(args.pred_b), keep, args.use_ab_only)
    yt, ya, yb, labels = _build_arrays(truth, pa, pb, args.drop_abstain)
    if len(yt) == 0:
        raise SystemExit("[paired] no overlapping records after filtering.")

    rare_fams = [x.strip() for x in args.rare_families.split(",") if x.strip()]
    m_a, r_a = _scores(yt, ya, labels, rare_fams)
    m_b, r_b = _scores(yt, yb, labels, rare_fams)
    d_macro = m_a - m_b
    d_rare = r_a - r_b

    rng = np.random.default_rng(args.seed)
    n = len(yt)
    dm = []
    dr = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        ma, ra = _scores(yt[idx], ya[idx], labels, rare_fams)
        mb, rb = _scores(yt[idx], yb[idx], labels, rare_fams)
        dm.append(ma - mb)
        dr.append(ra - rb)
    dm = np.asarray(dm)
    dr = np.asarray(dr)

    # Two-sided paired bootstrap p-value via sign reversal count.
    p_macro = float(2.0 * min((dm <= 0).mean(), (dm >= 0).mean()))
    p_rare = float(2.0 * min((dr <= 0).mean(), (dr >= 0).mean()))

    out = {
        "n_records": int(n),
        "name_a": args.name_a,
        "name_b": args.name_b,
        "macro_f1": {
            "a": m_a,
            "b": m_b,
            "delta_a_minus_b": d_macro,
            "ci95": [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))],
            "p_two_sided": p_macro,
        },
        "rare_f1": {
            "a": r_a,
            "b": r_b,
            "delta_a_minus_b": d_rare,
            "ci95": [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))],
            "p_two_sided": p_rare,
            "families": rare_fams,
        },
        "meta": {
            "drop_abstain": bool(args.drop_abstain),
            "use_ab_only": bool(args.use_ab_only),
            "manifest_jsonl": args.manifest_jsonl,
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
        },
    }

    print(
        f"[paired] n={n:,}  {args.name_a} vs {args.name_b}\n"
        f"  macro-F1: {m_a:.4f} vs {m_b:.4f}  delta={d_macro:+.4f}  "
        f"CI95=[{out['macro_f1']['ci95'][0]:+.4f},{out['macro_f1']['ci95'][1]:+.4f}]  p={p_macro:.4g}\n"
        f"  rare-F1 : {r_a:.4f} vs {r_b:.4f}  delta={d_rare:+.4f}  "
        f"CI95=[{out['rare_f1']['ci95'][0]:+.4f},{out['rare_f1']['ci95'][1]:+.4f}]  p={p_rare:.4g}"
    )

    if args.output:
        op = Path(args.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2) + "\n")
        print(f"[paired] wrote {op}")


if __name__ == "__main__":
    main()
