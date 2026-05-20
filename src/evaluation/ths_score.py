"""Tiered Hierarchy Score (THS): partial credit for getting deeper levels
of the hierarchical label right, even when the family is wrong.

Tier weights (sum = 1.0):
  family       0.40
  subtype      0.30
  direction    0.15
  polarity     0.15

A baseline that only predicts family caps out at 0.40 THS. Our LLM can
potentially hit 1.0. This is a *capability* metric, not just an accuracy
metric: it directly quantifies the hierarchical-prediction contribution.
"""
from __future__ import annotations
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

WEIGHTS = {"family": 0.40, "subtype": 0.30, "direction": 0.15, "polarity": 0.15}

SPLITS = {
    "random_full": dict(
        manifest=ROOT / "outputs/eval_prompts/random_full_test_5000_stratified.manifest.jsonl",
        llm=ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank8_abba.jsonl",
        llm_stack=ROOT / "outputs/eval_prompts/pred_cpu_stack_random_full.jsonl",
        mlp=ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_random_full_5k.jsonl",
        xgb=ROOT / "outputs/baselines_perpair/preds_xgb_random_full.jsonl",
    ),
    "drug_cold": dict(
        manifest=ROOT / "outputs/eval_prompts/drug_cold_test_5000_stratified.manifest.jsonl",
        llm=ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_drug_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
        llm_stack=ROOT / "outputs/eval_prompts/pred_cpu_stack_drug_cold.jsonl",
        mlp=ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_drug_cold_5k.jsonl",
        xgb=ROOT / "outputs/baselines_perpair/preds_xgb_drug_cold.jsonl",
    ),
    "pair_cold": dict(
        manifest=ROOT / "outputs/eval_prompts/pair_cold_test_5000_stratified.manifest.jsonl",
        llm=ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_pair_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
        llm_stack=ROOT / "outputs/eval_prompts/pred_cpu_stack_pair_cold.jsonl",
        mlp=ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_pair_cold_5k.jsonl",
        xgb=ROOT / "outputs/baselines_perpair/preds_xgb_pair_cold.jsonl",
    ),
}


def _truth_rich():
    rows = pq.read_table(
        ROOT / "data_processed/labels_hierarchical.parquet",
        columns=["pair_id", "family", "subtype", "polarity", "bidirectional"],
    ).to_pylist()
    out = {}
    for r in rows:
        out[r["pair_id"]] = {
            "family": r["family"],
            "subtype": r["subtype"],
            "polarity": r["polarity"],
            "direction": "bidirectional" if r["bidirectional"] else "directed",
        }
    return out


def _manifest(path):
    out = set()
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["pair_id"])
    return out


def _load(path, keep):
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("pair_id")
            if pid not in keep:
                continue
            if (r.get("input_order") or "ab").lower() != "ab":
                continue
            if pid in out:
                continue
            fp = r.get("final_prediction") or {}
            out[pid] = {
                "family": fp.get("family"),
                "subtype": fp.get("subtype"),
                "direction_tag": fp.get("direction_tag"),
                "polarity": fp.get("polarity"),
                "abstain": bool(fp.get("abstain", False)),
            }
    return out


def _norm_dir(d):
    if not d:
        return None
    d = str(d).lower()
    if "bidirect" in d:
        return "bidirectional"
    if d in {"a_to_b", "b_to_a", "directed"}:
        return "directed"
    return d


def _ths_one(pred, gold):
    if pred.get("abstain"):
        return 0.0
    s = 0.0
    if pred.get("family") and pred["family"] == gold["family"]:
        s += WEIGHTS["family"]
    if pred.get("subtype") and pred["subtype"] == gold["subtype"]:
        s += WEIGHTS["subtype"]
    if _norm_dir(pred.get("direction_tag")) == gold["direction"]:
        s += WEIGHTS["direction"]
    if pred.get("polarity") and pred["polarity"] == gold["polarity"]:
        s += WEIGHTS["polarity"]
    return s


def main():
    truth = _truth_rich()
    out_dir = ROOT / "outputs/diag2/headline"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    print(f"{'split':12s} {'model':12s} {'n':>6s} "
          f"{'THS':>8s} {'fam_acc':>8s} {'sub_acc':>8s} {'dir_acc':>8s} {'pol_acc':>8s}")
    print("-" * 88)

    for split, paths in SPLITS.items():
        keep = _manifest(paths["manifest"])
        if not keep:
            continue
        for model in ["llm", "llm_stack", "mlp", "xgb"]:
            src = _load(paths[model], keep)
            common = [pid for pid in keep if pid in src and pid in truth]
            if not common:
                continue
            ths_vals = []
            fam_ok = sub_ok = dir_ok = pol_ok = 0
            for pid in common:
                p = src[pid]
                g = truth[pid]
                ths_vals.append(_ths_one(p, g))
                if not p["abstain"]:
                    if p.get("family") and p["family"] == g["family"]:
                        fam_ok += 1
                    if p.get("subtype") and p["subtype"] == g["subtype"]:
                        sub_ok += 1
                    if _norm_dir(p.get("direction_tag")) == g["direction"]:
                        dir_ok += 1
                    if p.get("polarity") and p["polarity"] == g["polarity"]:
                        pol_ok += 1
            n = len(common)
            ths_mean = sum(ths_vals) / n
            summary.setdefault(split, {})[model] = {
                "n": n,
                "THS": ths_mean,
                "family_acc": fam_ok / n,
                "subtype_acc": sub_ok / n,
                "direction_acc": dir_ok / n,
                "polarity_acc": pol_ok / n,
            }
            print(f"{split:12s} {model:12s} {n:>6d} {ths_mean:>8.4f} "
                  f"{fam_ok/n:>8.4f} {sub_ok/n:>8.4f} {dir_ok/n:>8.4f} {pol_ok/n:>8.4f}")

    out = out_dir / "ths_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
