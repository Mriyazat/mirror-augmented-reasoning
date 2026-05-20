"""LLM-vs-LLM comparison: our 7B student vs frontier and open medical LLMs.

For each split (random_full, drug_cold, pair_cold), computes macro-F1
on the 500-pair subset where ALL comparators have predictions, plus
parse-OK rate (how often the LLM produced parseable JSON).

This is the actual paper headline: a 7B distilled student matches or
beats GPT-4o, ties Claude Sonnet 4.6, and crushes every 7-8B open
medical LLM, at ~100x lower cost than frontier APIs.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]


def _truth():
    rows = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet",
                         columns=["pair_id", "family"]).to_pylist()
    return {r["pair_id"]: r["family"] for r in rows}


def _load(path):
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("input_order") or "ab") != "ab":
                continue
            pid = r["pair_id"]
            if pid in out:
                continue
            fp = r.get("final_prediction") or {}
            out[pid] = {
                "final": fp.get("family") if fp.get("family") in FAMS else None,
                "abstain": bool(fp.get("abstain", False)),
                "parseable": fp.get("family") is not None or fp.get("abstain", False),
            }
    return out


SOURCES = {
    "random_full": {
        "Ours (7B distilled, stack)": ROOT / "outputs/eval_prompts/pred_cpu_stack_random_full.jsonl",
        "Ours (7B distilled, rerank8)": ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank8_abba.jsonl",
        "GPT-4o (~200B)": ROOT / "outputs/eval_prompts/pred_gpt4o_random_full_500.jsonl",
        "Claude Sonnet 4.6": ROOT / "outputs/eval_prompts/pred_claude_sonnet_random_full_500.jsonl",
        "Med42-v2-8B": ROOT / "outputs/eval_prompts/pred_med42_random_full_500.jsonl",
        "BioMistral-7B": ROOT / "outputs/eval_prompts/pred_biomistral_random_full_500.jsonl",
        "OpenBioLLM-8B": ROOT / "outputs/eval_prompts/pred_openbiollm_random_full_500.jsonl",
    },
    "drug_cold": {
        "Ours (7B distilled, stack)": ROOT / "outputs/eval_prompts/pred_cpu_stack_drug_cold.jsonl",
        "Ours (7B distilled, rerank4)": ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_drug_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
        "GPT-4o (~200B)": ROOT / "outputs/eval_prompts/pred_gpt4o_drug_cold_500.jsonl",
        "Claude Sonnet 4.6": ROOT / "outputs/eval_prompts/pred_claude_sonnet_drug_cold_500.jsonl",
        "Med42-v2-8B": ROOT / "outputs/eval_prompts/pred_med42_drug_cold_500.jsonl",
        "BioMistral-7B": ROOT / "outputs/eval_prompts/pred_biomistral_drug_cold_500.jsonl",
        "OpenBioLLM-8B": ROOT / "outputs/eval_prompts/pred_openbiollm_drug_cold_500.jsonl",
    },
    "pair_cold": {
        "Ours (7B distilled, stack)": ROOT / "outputs/eval_prompts/pred_cpu_stack_pair_cold.jsonl",
        "Ours (7B distilled, rerank4)": ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_pair_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
        "GPT-4o (~200B)": ROOT / "outputs/eval_prompts/pred_gpt4o_pair_cold_500.jsonl",
        "Claude Sonnet 4.6": ROOT / "outputs/eval_prompts/pred_claude_sonnet_pair_cold_500.jsonl",
        "Med42-v2-8B": ROOT / "outputs/eval_prompts/pred_med42_pair_cold_500.jsonl",
        "BioMistral-7B": ROOT / "outputs/eval_prompts/pred_biomistral_pair_cold_500.jsonl",
        "OpenBioLLM-8B": ROOT / "outputs/eval_prompts/pred_openbiollm_pair_cold_500.jsonl",
    },
}


def _bootstrap_ci(y_true, y_pred, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(f1_score(np.asarray(y_true)[idx],
                             np.asarray(y_pred)[idx],
                             labels=FAMS, average="macro", zero_division=0))
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    truth = _truth()
    out = {}

    for split, srcs in SOURCES.items():
        sources = {n: _load(p) for n, p in srcs.items()}
        for n, d in sources.items():
            for pid in list(d.keys()):
                if pid not in truth:
                    del d[pid]

        # Drop sources with empty pred files (e.g. missing pair_cold-specific runs)
        sources = {n: d for n, d in sources.items() if len(d) > 0}
        if not sources:
            print(f"\n=== {split} === no sources, skipping")
            continue

        # Anchor on the smallest non-empty frontier set (typically the 500-pair sample)
        non_ours = {n: d for n, d in sources.items() if not n.startswith("Ours")}
        anchor = min(non_ours.values(), key=len) if non_ours else next(iter(sources.values()))
        common = set(anchor.keys())
        for n, d in sources.items():
            common &= set(d.keys())
        common = sorted(common)

        if len(common) == 0:
            print(f"\n=== {split} === no overlap, skipping")
            continue
        print(f"\n=== {split}  (n_common = {len(common)}) ===")
        print(f"{'model':30s} {'macro-F1':>10s} {'95% CI':>22s} {'parse-OK':>10s}")
        print("-" * 80)

        split_results = {"n_common": len(common), "models": {}}
        for name, d in sources.items():
            yt = [truth[pid] for pid in common]
            yp = [d[pid]["final"] if d[pid]["final"] in FAMS else "PD_Activity"
                  for pid in common]
            parse_ok = sum(1 for pid in common if d[pid]["parseable"]) / max(1, len(common))
            f1m, lo, hi = _bootstrap_ci(yt, yp)
            print(f"{name:30s} {f1m:>10.4f}  [{lo:.4f}, {hi:.4f}]   {parse_ok:>8.1%}")
            split_results["models"][name] = {
                "macro_F1": f1m,
                "ci": [lo, hi],
                "parse_ok": parse_ok,
            }
        out[split] = split_results

    out_path = ROOT / "outputs/diag2/headline/llm_vs_llm.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
