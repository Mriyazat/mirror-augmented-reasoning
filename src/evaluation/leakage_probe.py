"""Anti-memorisation probe.

For each test pair compute the *rarer-endpoint train coverage* = min over
the two drugs of the number of training pairs that include it. Plot mean
per-pair accuracy vs train-coverage bin for LLM / MLP / XGB.

A purely memorising model will see accuracy collapse on the
low-coverage end. A generalising model will look flat.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "outputs/figures"
FIGS.mkdir(parents=True, exist_ok=True)


def _split_drug_coverage(split: str) -> dict[str, int]:
    """For each drug, count number of train pairs involving it (the split's train)."""
    manifest = pq.read_table(ROOT / f"data_processed/splits/manifest_{split}.parquet").to_pandas()
    pairs = pq.read_table(ROOT / "data_processed/pairs.parquet", columns=["pair_id", "a_id", "b_id"]).to_pandas()
    train_ids = set(manifest[manifest["split"] == "train"]["pair_id"].tolist())
    cov = defaultdict(int)
    for r in pairs.itertuples():
        if r.pair_id in train_ids:
            cov[r.a_id] += 1
            cov[r.b_id] += 1
    return dict(cov)


def _pair_lookup() -> dict[str, tuple[str, str]]:
    p = pq.read_table(ROOT / "data_processed/pairs.parquet", columns=["pair_id", "a_id", "b_id"]).to_pandas()
    return {r.pair_id: (r.a_id, r.b_id) for r in p.itertuples()}


def _truth_lookup() -> dict[str, str]:
    t = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet", columns=["pair_id", "family"]).to_pandas()
    return {r.pair_id: r.family for r in t.itertuples()}


def _load(path: Path) -> dict[str, dict]:
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
            if (r.get("input_order") or "ab").lower() != "ab":
                continue
            if pid in out:
                continue
            fp = r.get("final_prediction") or {}
            out[pid] = {"family": fp.get("family"), "abstain": bool(fp.get("abstain", False))}
    return out


def _manifest(path: Path) -> list[str]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)["pair_id"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=6)
    args = ap.parse_args()

    splits = {
        "random_full": dict(
            man=ROOT / "outputs/eval_prompts/random_full_test_5000_stratified.manifest.jsonl",
            llm=ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank4_abba.jsonl",
            mlp=ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_random_full_5k.jsonl",
            xgb=ROOT / "outputs/baselines_perpair/preds_xgb_random_full.jsonl",
        ),
        "drug_cold": dict(
            man=ROOT / "outputs/eval_prompts/drug_cold_test_5000_stratified.manifest.jsonl",
            llm=ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_drug_cold_test_5000_stratified_nb_abba.jsonl",
            mlp=ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_drug_cold_5k.jsonl",
            xgb=ROOT / "outputs/baselines_perpair/preds_xgb_drug_cold.jsonl",
        ),
        "pair_cold": dict(
            man=ROOT / "outputs/eval_prompts/pair_cold_test_5000_stratified.manifest.jsonl",
            llm=ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_pair_cold_test_5000_stratified_nb_abba.jsonl",
            mlp=ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_pair_cold_5k.jsonl",
            xgb=ROOT / "outputs/baselines_perpair/preds_xgb_pair_cold.jsonl",
        ),
    }

    truth = _truth_lookup()
    pairs = _pair_lookup()

    # Coverage definition per split:
    #   random_full: min(cov(a), cov(b)) — rarer-endpoint memorisation probe.
    #   drug_cold:   max(cov(a), cov(b)) — warmer-side leakage probe; rarer side is 0.
    #   pair_cold:   not available (both sides have 0 train coverage); we use
    #                Tanimoto similarity to the closest training drug instead.
    fig, axes = plt.subplots(1, len(splits), figsize=(5.5 * len(splits), 5), sharey=True)
    rho_table = {}
    for ax, (sp, paths) in zip(axes, splits.items()):
        cov = _split_drug_coverage(sp)
        pids = _manifest(paths["man"])
        cov_x = []
        keep = []
        for pid in pids:
            ab = pairs.get(pid)
            if not ab:
                continue
            if sp == "random_full":
                v = min(cov.get(ab[0], 0), cov.get(ab[1], 0))
            elif sp == "drug_cold":
                v = max(cov.get(ab[0], 0), cov.get(ab[1], 0))
            elif sp == "pair_cold":
                v = cov.get(ab[0], 0) + cov.get(ab[1], 0)
            else:
                v = 0
            cov_x.append(v)
            keep.append(pid)
        cov_x = np.asarray(cov_x)
        if cov_x.size == 0 or cov_x.std() == 0:
            ax.set_title(f"{sp} (probe N/A: zero coverage variance)")
            continue
        edges = np.unique(np.quantile(cov_x, np.linspace(0, 1, args.bins + 1)))
        mids = 0.5 * (edges[:-1] + edges[1:])

        rho_table[sp] = {}
        colors = {"LLM": "#d62728", "MLP": "#1f77b4", "XGB": "#2ca02c"}
        for name, key in [("LLM", "llm"), ("MLP", "mlp"), ("XGB", "xgb")]:
            src = _load(paths[key])
            common = [pid for pid in keep if pid in src and pid in truth]
            if sp == "random_full":
                cov_c = np.asarray([min(cov.get(pairs[pid][0], 0), cov.get(pairs[pid][1], 0)) for pid in common])
            elif sp == "drug_cold":
                cov_c = np.asarray([max(cov.get(pairs[pid][0], 0), cov.get(pairs[pid][1], 0)) for pid in common])
            else:
                cov_c = np.asarray([cov.get(pairs[pid][0], 0) + cov.get(pairs[pid][1], 0) for pid in common])
            correct = np.asarray([int(src[pid]["family"] == truth[pid] and not src[pid]["abstain"]) for pid in common])

            if cov_c.size > 10 and correct.std() > 0 and cov_c.std() > 0:
                from scipy.stats import spearmanr
                rho, p = spearmanr(cov_c, correct)
                rho_table[sp][name] = {"spearman_rho": float(rho), "p": float(p)}

            bin_means = []
            bin_xs = []
            for i in range(len(edges) - 1):
                m = (cov_c >= edges[i]) & (cov_c < edges[i + 1])
                if i == len(edges) - 2:
                    m = (cov_c >= edges[i]) & (cov_c <= edges[i + 1])
                if m.sum() >= 30:
                    bin_means.append(correct[m].mean())
                    bin_xs.append(mids[i])
            if bin_xs:
                rho_str = ""
                if name in rho_table.get(sp, {}):
                    rho_str = f" (ρ={rho_table[sp][name]['spearman_rho']:+.2f})"
                ax.plot(bin_xs, bin_means, "o-", color=colors[name], label=f"{name}{rho_str}")

        xlab = {
            "random_full": "min(cov(a), cov(b)) — rarer-endpoint train pairs",
            "drug_cold":   "max(cov(a), cov(b)) — warmer-endpoint train pairs",
            "pair_cold":   "cov(a) + cov(b) — total endpoint train pairs",
        }[sp]
        ax.set_title(sp)
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel("Mean per-pair accuracy")
        ax.set_xscale("symlog")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Anti-memorisation probe: accuracy vs rarer-endpoint train coverage")
    fig.tight_layout()
    out = FIGS / "fig_leakage_probe.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    (ROOT / "outputs/diag2/headline").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/diag2/headline/leakage_rho.json").write_text(json.dumps(rho_table, indent=2) + "\n")
    print(f"[fig] {out}")
    print(json.dumps(rho_table, indent=2))


if __name__ == "__main__":
    main()
