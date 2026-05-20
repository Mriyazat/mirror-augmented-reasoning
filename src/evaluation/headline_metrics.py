"""Compute the full headline metric surface for every model and split.

For each (split, model) emits, into outputs/diag2/headline/<split>.json:
    macro_F1, weighted_F1, macro_F1_rare, n_predicted, n_abstain,
    per_family_F1 (with 95% bootstrap CI), selective_acc@{50,70,90} (LLM only),
    subtype_acc (LLM only, conditional on family correct),
    subtype_macro_F1 (LLM only, over subtypes seen in test).

Also produces:
    outputs/diag2/headline/MASTER_TABLE.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
RARE_FAMILIES = {"PK_Absorption", "PK_Distribution", "PK_Excretion"}


SPLITS = {
    "random_full": dict(
        manifest=ROOT / "outputs/eval_prompts/random_full_test_5000_stratified.manifest.jsonl",
        llm=ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank8_abba.jsonl",
        llm_stack=ROOT / "outputs/eval_prompts/pred_cpu_stack_random_full.jsonl",
        mlp=ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_random_full_5k.jsonl",
        xgb=ROOT / "outputs/baselines_perpair/preds_xgb_random_full.jsonl",
        logreg=ROOT / "outputs/baselines_perpair/preds_logreg_fast_random_full_5k.jsonl",
    ),
    "drug_cold": dict(
        manifest=ROOT / "outputs/eval_prompts/drug_cold_test_5000_stratified.manifest.jsonl",
        llm=ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_drug_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
        llm_stack=ROOT / "outputs/eval_prompts/pred_cpu_stack_drug_cold.jsonl",
        mlp=ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_drug_cold_5k.jsonl",
        xgb=ROOT / "outputs/baselines_perpair/preds_xgb_drug_cold.jsonl",
        logreg=ROOT / "outputs/baselines_perpair/preds_logreg_fast_drug_cold_5k.jsonl",
    ),
    "pair_cold": dict(
        manifest=ROOT / "outputs/eval_prompts/pair_cold_test_5000_stratified.manifest.jsonl",
        llm=ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_pair_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
        llm_stack=ROOT / "outputs/eval_prompts/pred_cpu_stack_pair_cold.jsonl",
        mlp=ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_pair_cold_5k.jsonl",
        xgb=ROOT / "outputs/baselines_perpair/preds_xgb_pair_cold.jsonl",
        logreg=ROOT / "outputs/baselines_perpair/preds_logreg_fast_pair_cold_5k.jsonl",
    ),
}


def _read_truth():
    t = pq.read_table(
        ROOT / "data_processed/labels_hierarchical.parquet",
        columns=["pair_id", "family", "subtype"],
    ).to_pylist()
    fam = {r["pair_id"]: r["family"] for r in t}
    sub = {r["pair_id"]: r["subtype"] for r in t}
    return fam, sub


def _manifest(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["pair_id"])
    return out


def _load(path: Path, keep: set[str]):
    """Return {pair_id -> {'family','subtype','abstain','conf'}}"""
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("pair_id")
            if pid not in keep:
                continue
            if (r.get("input_order") or "ab").lower() != "ab":
                continue
            if pid in out:
                continue
            fp = r.get("final_prediction") or {}
            try:
                conf = float(fp.get("confidence")) if fp.get("confidence") is not None else None
            except Exception:
                conf = None
            out[pid] = {
                "family": fp.get("family"),
                "subtype": fp.get("subtype"),
                "abstain": bool(fp.get("abstain", False)),
                "conf": conf,
            }
    return out


def _arrays(pids, truth_fam, preds, lab2idx):
    sentinel = len(lab2idx)
    yt = np.asarray([lab2idx[truth_fam[pid]] for pid in pids])
    yp = np.asarray(
        [
            sentinel if (preds[pid]["abstain"] or preds[pid]["family"] not in lab2idx)
            else lab2idx[preds[pid]["family"]]
            for pid in pids
        ]
    )
    return yt, yp


def _macro(yt, yp, n):
    return float(f1_score(yt, yp, labels=list(range(n)), average="macro", zero_division=0))


def _rare_macro(yt, yp, lab2idx):
    rare_ids = [lab2idx[f] for f in RARE_FAMILIES if f in lab2idx]
    return float(f1_score(yt, yp, labels=rare_ids, average="macro", zero_division=0))


def _selective(yt, yp_full, conf, threshold):
    """Fraction of (above-threshold and correct) over above-threshold."""
    mask = (conf >= threshold) & (yp_full != -1)
    if mask.sum() == 0:
        return None, 0
    correct = (yp_full[mask] == yt[mask]).mean()
    return float(correct), int(mask.sum())


def _per_family_ci(yt, yp, labels, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    n_classes = len(labels)
    base = f1_score(yt, yp, labels=list(range(n_classes)), average=None, zero_division=0)
    boots = np.zeros((n_boot, n_classes))
    n = len(yt)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = f1_score(yt[idx], yp[idx], labels=list(range(n_classes)), average=None, zero_division=0)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return {labels[i]: {"f1": float(base[i]), "ci95": [float(lo[i]), float(hi[i])]} for i in range(n_classes)}


def main():
    truth_fam, truth_sub = _read_truth()
    families = sorted(set(truth_fam.values()))
    lab2idx = {f: i for i, f in enumerate(families)}
    n_classes = len(families)
    out_dir = ROOT / "outputs/diag2/headline"
    out_dir.mkdir(parents=True, exist_ok=True)

    master_rows = []

    for sp, paths in SPLITS.items():
        keep = _manifest(paths["manifest"])
        if not keep:
            print(f"[warn] no manifest for {sp}")
            continue
        sources = {
            name: _load(paths[key], keep)
            for name, key in [
                ("LLM", "llm"),
                ("LLM_stack", "llm_stack"),
                ("MLP", "mlp"),
                ("XGB", "xgb"),
                ("LogReg", "logreg"),
            ]
        }
        common = sorted(set(keep) & set(truth_fam))
        for name, src in sources.items():
            common = [pid for pid in common if pid in src]
        print(f"[hd] {sp} common={len(common)}")
        split_out = {"split": sp, "n_common": len(common), "models": {}}

        for name, src in sources.items():
            yt, yp = _arrays(common, truth_fam, src, lab2idx)
            macro = _macro(yt, yp, n_classes)
            rare = _rare_macro(yt, yp, lab2idx)
            weighted = float(
                f1_score(yt, yp, labels=list(range(n_classes)), average="weighted", zero_division=0)
            )
            n_abstain = int((yp == n_classes).sum())
            per_family = _per_family_ci(yt, yp, families, n_boot=400, seed=1)

            entry = {
                "macro_F1": macro,
                "weighted_F1": weighted,
                "rare_macro_F1": rare,
                "n_test": int(len(yt)),
                "n_abstain": n_abstain,
                "per_family": per_family,
            }

            if name in ("LLM", "LLM_stack"):
                confs = np.asarray([src[pid]["conf"] if src[pid]["conf"] is not None else 0.0 for pid in common])
                yp_full = yp.copy()
                yp_full[yp == n_classes] = -1
                sel = {}
                for thr in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85]:
                    acc, cov = _selective(yt, yp_full, confs, thr)
                    sel[f"acc@{thr:.2f}"] = {"accuracy": acc, "n_covered": cov, "coverage": cov / max(1, len(yt))}
                entry["selective"] = sel

                pairs_with_sub = [pid for pid in common if src[pid]["subtype"] and not src[pid]["abstain"]]
                if pairs_with_sub:
                    sub_correct = sum(
                        1 for pid in pairs_with_sub
                        if src[pid]["subtype"] == truth_sub.get(pid)
                    )
                    sub_acc = sub_correct / len(pairs_with_sub)
                    fam_correct_ids = [
                        pid for pid in pairs_with_sub
                        if src[pid]["family"] == truth_fam.get(pid)
                    ]
                    sub_given_fam = (
                        sum(
                            1 for pid in fam_correct_ids
                            if src[pid]["subtype"] == truth_sub.get(pid)
                        ) / max(1, len(fam_correct_ids))
                    )
                    pred_subtypes = [src[pid]["subtype"] for pid in pairs_with_sub]
                    gold_subtypes = [truth_sub.get(pid, "UNK") for pid in pairs_with_sub]
                    sub_labels = sorted(set(pred_subtypes) | set(gold_subtypes))
                    s2i = {s: i for i, s in enumerate(sub_labels)}
                    ys_t = np.asarray([s2i[s] for s in gold_subtypes])
                    ys_p = np.asarray([s2i[s] for s in pred_subtypes])
                    sub_macro = float(f1_score(ys_t, ys_p, labels=list(range(len(sub_labels))), average="macro", zero_division=0))
                    sub_weighted = float(f1_score(ys_t, ys_p, labels=list(range(len(sub_labels))), average="weighted", zero_division=0))
                    entry["subtype"] = {
                        "n_subtype_unique_in_test": len(sub_labels),
                        "subtype_accuracy": sub_acc,
                        "subtype_accuracy_given_family_correct": sub_given_fam,
                        "subtype_macro_F1": sub_macro,
                        "subtype_weighted_F1": sub_weighted,
                    }

            split_out["models"][name] = entry
            llm_like = name in ("LLM", "LLM_stack")
            row = {
                "split": sp,
                "model": name,
                "n_test": entry["n_test"],
                "n_abstain": entry["n_abstain"],
                "macro_F1": round(entry["macro_F1"], 4),
                "rare_macro_F1": round(entry["rare_macro_F1"], 4),
                "weighted_F1": round(entry["weighted_F1"], 4),
                "subtype_acc": round(entry["subtype"]["subtype_accuracy"], 4) if llm_like and "subtype" in entry else "",
                "subtype_acc_given_fam": round(entry["subtype"]["subtype_accuracy_given_family_correct"], 4) if llm_like and "subtype" in entry else "",
                "subtype_macroF1": round(entry["subtype"]["subtype_macro_F1"], 4) if llm_like and "subtype" in entry else "",
                "sel_acc@0.70": round(entry["selective"]["acc@0.70"]["accuracy"] or 0.0, 4) if llm_like else "",
                "sel_cov@0.70": round(entry["selective"]["acc@0.70"]["coverage"] or 0.0, 4) if llm_like else "",
                "sel_acc@0.80": round(entry["selective"]["acc@0.80"]["accuracy"] or 0.0, 4) if llm_like else "",
                "sel_cov@0.80": round(entry["selective"]["acc@0.80"]["coverage"] or 0.0, 4) if llm_like else "",
                "sel_acc@0.85": round(entry["selective"]["acc@0.85"]["accuracy"] or 0.0, 4) if llm_like else "",
                "sel_cov@0.85": round(entry["selective"]["acc@0.85"]["coverage"] or 0.0, 4) if llm_like else "",
            }
            master_rows.append(row)

        (out_dir / f"{sp}.json").write_text(json.dumps(split_out, indent=2) + "\n")
        print(f"[hd] wrote {out_dir / (sp + '.json')}")

    csv_path = out_dir / "MASTER_TABLE.csv"
    if master_rows:
        keys = list(master_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in master_rows:
                w.writerow(r)
        print(f"[hd] wrote {csv_path}")


if __name__ == "__main__":
    main()
