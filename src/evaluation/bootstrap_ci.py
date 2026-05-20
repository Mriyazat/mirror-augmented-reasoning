"""Bootstrap 95% CIs for family macro-F1 + macro-THS, with three
abstention policies:
  wrong  -> abstained predictions count as a wrong sentinel class
  keep   -> abstained predictions are scored at their predicted family
  drop   -> abstained predictions are removed (selective F1)

For selective methods (prm_vote_consensus, vote_margin_score) you almost
always want --drop_abstain to report the selective accuracy.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]


def _load_predictions(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("pair_id")
            if not pid:
                continue
            order = (rec.get("input_order") or "ab").lower()
            fp = rec.get("final_prediction") or {}
            out[pid][f"{order}_family"] = fp.get("family")
            out[pid][f"{order}_abstain"] = bool(fp.get("abstain", False))
    return dict(out)


def _load_truth(labels_path: Path, manifest_pids: set[str] | None) -> dict[str, str]:
    rows = pq.read_table(labels_path, columns=["pair_id", "family"]).to_pylist()
    out = {r["pair_id"]: r["family"] for r in rows}
    if manifest_pids is not None:
        out = {pid: f for pid, f in out.items() if pid in manifest_pids}
    return out


def _load_manifest_pids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    pids: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pids.add(json.loads(line)["pair_id"])
    return pids


def _per_record(preds, truth, orderings, policy):
    labels = sorted(set(truth.values()))
    if not labels:
        raise SystemExit("No families found.")
    sentinel = "__abstain__"
    lab2idx = {f: i for i, f in enumerate(labels)}
    lab2idx[sentinel] = len(labels)
    yt, yp = [], []
    n_dropped = 0
    for pid, rec in preds.items():
        gold = truth.get(pid)
        if gold is None:
            continue
        for o in orderings:
            fkey = f"{o}_family"
            akey = f"{o}_abstain"
            if fkey not in rec:
                continue
            f_pred = rec[fkey]
            abstained = rec.get(akey, False)
            if policy == "drop" and abstained:
                n_dropped += 1
                continue
            if f_pred is None:
                idx = lab2idx[sentinel]
            elif policy == "wrong" and abstained:
                idx = lab2idx[sentinel]
            elif f_pred not in lab2idx:
                idx = lab2idx[sentinel]
            else:
                idx = lab2idx[f_pred]
            yt.append(lab2idx[gold])
            yp.append(idx)
    return np.asarray(yt), np.asarray(yp), labels, n_dropped


def _macro_f1(yt, yp, labels_int):
    return float(f1_score(yt, yp, labels=labels_int, average="macro", zero_division=0))


def _ths_family_only(yt, yp, labels, abstain_idx):
    W = 0.3
    per_fam: dict[str, list[float]] = defaultdict(list)
    for t, p in zip(yt, yp):
        gold = labels[t]
        s = W if (p != abstain_idx and p < len(labels) and labels[p] == gold) else 0.0
        per_fam[gold].append(s)
    if not per_fam:
        return 0.0
    return float(np.mean([float(np.mean(v)) for v in per_fam.values()]))


def _bootstrap(yt, yp, labels, abstain_idx, n_boot, seed):
    rng = np.random.default_rng(seed)
    labels_int = list(range(len(labels)))
    base_f1 = _macro_f1(yt, yp, labels_int)
    base_ths = _ths_family_only(yt, yp, labels, abstain_idx)
    n = len(yt)
    fam_to_idx = {cls: np.where(yt == cls)[0] for cls in labels_int}
    sizes = {cls: len(fam_to_idx[cls]) for cls in labels_int}
    f1s, thss = [], []
    for _ in range(n_boot):
        idx_list = []
        for cls in labels_int:
            if sizes[cls] == 0:
                continue
            picks = rng.integers(0, sizes[cls], size=sizes[cls])
            idx_list.append(fam_to_idx[cls][picks])
        idx = np.concatenate(idx_list)
        f1s.append(_macro_f1(yt[idx], yp[idx], labels_int))
        thss.append(_ths_family_only(yt[idx], yp[idx], labels, abstain_idx))
    def _ci(a):
        arr = np.asarray(a)
        return {"mean": float(arr.mean()),
                "lo": float(np.percentile(arr, 2.5)),
                "hi": float(np.percentile(arr, 97.5))}
    return {
        "n_records": int(n),
        "macro_f1":  {"point": base_f1,  **_ci(f1s)},
        "macro_ths": {"point": base_ths, **_ci(thss)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--labels", default="data_processed/labels_hierarchical.parquet")
    ap.add_argument("--manifest_jsonl", default=None)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--use_ab_only", action="store_true")
    ap.add_argument("--keep_abstain", action="store_true")
    ap.add_argument("--drop_abstain", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.keep_abstain and args.drop_abstain:
        raise SystemExit("--keep_abstain and --drop_abstain are mutually exclusive.")
    policy = "keep" if args.keep_abstain else ("drop" if args.drop_abstain else "wrong")

    out_path = (Path(args.output) if args.output
                else ROOT / "outputs" / "audit" / f"bootstrap_{args.run_name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preds = _load_predictions(Path(args.predictions))
    manifest = _load_manifest_pids(Path(args.manifest_jsonl) if args.manifest_jsonl else None)
    truth = _load_truth(Path(args.labels), manifest)
    if manifest is not None:
        preds = {pid: r for pid, r in preds.items() if pid in manifest}

    orderings = ("ab",) if args.use_ab_only else ("ab", "ba")
    yt, yp, labels, n_dropped = _per_record(preds, truth, orderings, policy)
    if len(yt) == 0:
        raise SystemExit("[boot] no records to score.")
    coverage = len(yt) / (len(yt) + n_dropped) if (len(yt) + n_dropped) else 0.0
    print(f"[boot] {len(yt):,} records  policy={policy}  "
          f"n_dropped={n_dropped:,}  coverage={coverage:.3f}")
    result = _bootstrap(yt, yp, labels, len(labels), args.n_boot, args.seed)
    result["meta"] = {"predictions": str(args.predictions),
                      "policy": policy, "coverage": coverage,
                      "use_ab_only": bool(args.use_ab_only),
                      "n_dropped": int(n_dropped)}
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[boot] macro-F1 = {result['macro_f1']['point']:.4f}  "
          f"[{result['macro_f1']['lo']:.4f}, {result['macro_f1']['hi']:.4f}]  "
          f"(n={len(yt):,}, coverage={coverage:.3f})")
    print(f"[boot] macro-THS = {result['macro_ths']['point']:.4f}  "
          f"[{result['macro_ths']['lo']:.4f}, {result['macro_ths']['hi']:.4f}]")
    print(f"[boot] wrote {out_path}")


if __name__ == "__main__":
    main()
