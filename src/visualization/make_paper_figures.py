"""Generate paper figures as PNG images.

Produces:
  fig_confusion_<split>_<model>.png  (LLM, MLP, XGB on each split)
  fig_per_family_F1_by_split.png
  fig_generalization_drop.png
  fig_complementarity_pair_cold.png
  fig_dataset_family_distribution.png
  fig_dataset_drug_degree.png
  fig_dataset_split_sizes.png
  fig_calibration_llm_pair_cold.png

All written under outputs/figures/.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_processed"
FIGS = ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

FAMILIES = None  # populated in main


def _read_truth():
    rows = pq.read_table(
        DATA / "labels_hierarchical.parquet",
        columns=["pair_id", "family", "subtype"],
    ).to_pylist()
    truth = {r["pair_id"]: r["family"] for r in rows}
    return truth, sorted({r["family"] for r in rows})


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


def _load_preds(path: Path, keep: set[str]) -> dict[str, dict]:
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
                "abstain": bool(fp.get("abstain", False)),
                "conf": conf,
            }
    return out


def _arrays(common, truth, preds, lab2idx):
    sentinel = len(lab2idx)
    yt = np.asarray([lab2idx[truth[pid]] for pid in common])
    yp = np.asarray(
        [
            sentinel if (preds[pid]["abstain"] or preds[pid]["family"] not in lab2idx)
            else lab2idx[preds[pid]["family"]]
            for pid in common
        ]
    )
    return yt, yp


# ----------------------------------------------------------------------
# Plotters
# ----------------------------------------------------------------------
def plot_confusion(yt, yp, labels, title, out_path, normalize=True):
    full_labels = labels + ["__abstain__"]
    cm = confusion_matrix(yt, yp, labels=list(range(len(full_labels))))
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_n = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
    else:
        cm_n = cm.astype(float)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1 if normalize else cm_n.max())
    ax.set_xticks(range(len(full_labels)))
    ax.set_yticks(range(len(full_labels)))
    ax.set_xticklabels(full_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(full_labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    for i in range(len(full_labels)):
        for j in range(len(full_labels)):
            v = cm_n[i, j]
            if v > 0:
                txt = f"{v:.2f}" if normalize else f"{int(v)}"
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_family_by_split(per_split, labels, out_path):
    splits = list(per_split.keys())
    models = list(next(iter(per_split.values())).keys())
    n_fam = len(labels)
    x = np.arange(n_fam)
    width = 0.8 / len(models)
    fig, axes = plt.subplots(1, len(splits), figsize=(5.5 * len(splits), 5), sharey=True)
    if len(splits) == 1:
        axes = [axes]
    colors = {"LLM": "#d62728", "MLP": "#1f77b4", "XGB": "#2ca02c", "LogReg": "#ff7f0e"}
    for ax, sp in zip(axes, splits):
        models_here = list(per_split[sp].keys())
        for i, m in enumerate(models_here):
            f1s = per_split[sp][m]
            ax.bar(x + i * width - 0.4 + width / 2, f1s, width, label=m, color=colors.get(m, None))
        ax.set_title(sp)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Per-family macro-F1 by split (compulsory, AB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_generalization_drop(macro_by_split, out_path):
    splits = ["random_full", "drug_cold", "pair_cold"]
    models = list(next(iter(macro_by_split.values())).keys())
    x = np.arange(len(splits))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"LLM": "#d62728", "MLP": "#1f77b4", "XGB": "#2ca02c", "LogReg": "#ff7f0e", "Majority": "#7f7f7f"}
    for i, m in enumerate(models):
        vals = [macro_by_split[s][m] for s in splits]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=m, color=colors.get(m, None))
        for xi, v in enumerate(vals):
            ax.text(xi + i * width - 0.4 + width / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Generalization across splits (warm -> cold)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_complementarity(per_family_complement, labels, out_path, title):
    only_l = [per_family_complement[f]["only_llm"] for f in labels]
    only_b = [per_family_complement[f]["only_base"] for f in labels]
    both_c = [per_family_complement[f]["both_correct"] for f in labels]
    both_w = [per_family_complement[f]["both_wrong"] for f in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    p1 = ax.bar(x, both_c, label="both correct", color="#2ca02c")
    p2 = ax.bar(x, only_l, bottom=both_c, label="only LLM", color="#d62728")
    p3 = ax.bar(x, only_b, bottom=np.array(both_c) + np.array(only_l), label="only baseline", color="#1f77b4")
    p4 = ax.bar(
        x, both_w,
        bottom=np.array(both_c) + np.array(only_l) + np.array(only_b),
        label="both wrong", color="#7f7f7f",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Pairs")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_family_distribution(out_path):
    rows = pq.read_table(DATA / "labels_hierarchical.parquet", columns=["family"]).to_pylist()
    c = Counter(r["family"] for r in rows)
    labels = sorted(c.keys())
    vals = [c[l] for l in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, vals, color="#1f77b4")
    ax.set_title(f"Family distribution in this dataset (n={sum(vals):,})")
    ax.set_ylabel("Pairs")
    ax.tick_params(axis="x", rotation=30)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v/1000:.0f}k",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_drug_degree(out_path):
    pairs = pq.read_table(DATA / "pairs.parquet", columns=["a_id", "b_id"]).to_pylist()
    deg = Counter()
    for r in pairs:
        deg[r["a_id"]] += 1
        deg[r["b_id"]] += 1
    vals = np.asarray(list(deg.values()))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(vals, bins=80, color="#2ca02c")
    ax.set_yscale("log")
    ax.set_xlabel("Pairs per drug")
    ax.set_ylabel("Count of drugs (log)")
    ax.set_title(f"Drug interaction degree distribution (n_drugs={len(vals):,})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_split_sizes(out_path):
    splits = ["random_full", "drug_cold", "pair_cold"]
    sizes_train = []
    sizes_val = []
    sizes_test = []
    for sp in splits:
        m = pq.read_table(DATA / "splits" / f"manifest_{sp}.parquet").to_pylist()
        c = Counter(r["split"] for r in m)
        sizes_train.append(c.get("train", 0))
        sizes_val.append(c.get("val", 0))
        sizes_test.append(c.get("test", 0))
    x = np.arange(len(splits))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, sizes_train, width, label="train", color="#1f77b4")
    ax.bar(x, sizes_val, width, label="val", color="#ff7f0e")
    ax.bar(x + width, sizes_test, width, label="test", color="#d62728")
    for xi, (t, v, te) in enumerate(zip(sizes_train, sizes_val, sizes_test)):
        for off, val in zip([-width, 0, width], [t, v, te]):
            ax.text(xi + off, val, f"{val/1000:.0f}k", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Pairs")
    ax.set_title("Split sizes per protocol (full dataset)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_reliability(confs, correct, out_path, title):
    confs = np.asarray(confs, dtype=float)
    correct = np.asarray(correct, dtype=int)
    bins = np.linspace(0.0, 1.0, 11)
    mids, accs, counts = [], [], []
    for i in range(len(bins) - 1):
        m = (confs >= bins[i]) & (confs < bins[i + 1])
        if i == len(bins) - 2:
            m = (confs >= bins[i]) & (confs <= bins[i + 1])
        if m.sum() == 0:
            continue
        mids.append((bins[i] + bins[i + 1]) / 2)
        accs.append(correct[m].mean())
        counts.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
    ax.plot(mids, accs, "o-", color="#d62728", label="LLM observed")
    for x, y, n in zip(mids, accs, counts):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
SPLITS = {
    "random_full": dict(
        manifest=ROOT / "outputs/eval_prompts/random_full_test_5000_stratified.manifest.jsonl",
        llm=ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank4_abba.jsonl",
        mlp=ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_random_full_5k.jsonl",
        xgb=ROOT / "outputs/baselines_perpair/preds_xgb_random_full.jsonl",
        logreg=ROOT / "outputs/baselines_perpair/preds_logreg_fast_random_full_5k.jsonl",
    ),
    "drug_cold": dict(
        manifest=ROOT / "outputs/eval_prompts/drug_cold_test_5000_stratified.manifest.jsonl",
        llm=ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_drug_cold_test_5000_stratified_nb_abba.jsonl",
        mlp=ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_drug_cold_5k.jsonl",
        xgb=ROOT / "outputs/baselines_perpair/preds_xgb_drug_cold.jsonl",
        logreg=ROOT / "outputs/baselines_perpair/preds_logreg_fast_drug_cold_5k.jsonl",
    ),
    "pair_cold": dict(
        manifest=ROOT / "outputs/eval_prompts/pair_cold_test_5000_stratified.manifest.jsonl",
        llm=ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_pair_cold_test_5000_stratified_nb_abba.jsonl",
        mlp=ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_pair_cold_5k.jsonl",
        xgb=ROOT / "outputs/baselines_perpair/preds_xgb_pair_cold.jsonl",
        logreg=ROOT / "outputs/baselines_perpair/preds_logreg_fast_pair_cold_5k.jsonl",
    ),
}


def main():
    truth, labels = _read_truth()
    lab2idx = {f: i for i, f in enumerate(labels)}

    macro_by_split: dict[str, dict[str, float]] = {}
    per_family_by_split: dict[str, dict[str, list[float]]] = {}
    pair_cold_complement = None

    for sp, paths in SPLITS.items():
        keep = _manifest(paths["manifest"])
        if not keep:
            print(f"[warn] missing manifest for {sp}")
            continue
        llm = _load_preds(paths["llm"], keep)
        mlp = _load_preds(paths["mlp"], keep)
        xgb = _load_preds(paths["xgb"], keep)
        logreg = _load_preds(paths["logreg"], keep)
        common = sorted(set(keep) & set(truth) & set(mlp) & set(xgb))
        if not common:
            print(f"[warn] no common predictions for {sp}")
            continue
        if llm:
            common = [pid for pid in common if pid in llm]
        n_classes = len(labels)
        macro_by_split[sp] = {}
        per_family_by_split[sp] = {}
        for name, src in [("LLM", llm), ("MLP", mlp), ("XGB", xgb), ("LogReg", logreg)]:
            if not src:
                continue
            yt, yp = _arrays(common, truth, src, lab2idx)
            macro = float(
                f1_score(yt, yp, labels=list(range(n_classes)), average="macro", zero_division=0)
            )
            per_fam = f1_score(yt, yp, labels=list(range(n_classes)), average=None, zero_division=0)
            macro_by_split[sp][name] = macro
            per_family_by_split[sp][name] = list(per_fam)
            plot_confusion(
                yt, yp, labels,
                title=f"Confusion - {name} on {sp}",
                out_path=FIGS / f"fig_confusion_{sp}_{name}.png",
            )
            print(f"[fig] confusion {sp}/{name} -> macro={macro:.4f}")

        if sp == "pair_cold" and "LLM" in macro_by_split[sp] and "XGB" in macro_by_split[sp]:
            yt, ylm = _arrays(common, truth, llm, lab2idx)
            _, yxg = _arrays(common, truth, xgb, lab2idx)
            comp = {}
            for fam in labels:
                fi = lab2idx[fam]
                mask = yt == fi
                comp[fam] = {
                    "only_llm": int(((ylm == fi) & (yxg != fi) & mask).sum()),
                    "only_base": int(((yxg == fi) & (ylm != fi) & mask).sum()),
                    "both_correct": int(((ylm == fi) & (yxg == fi) & mask).sum()),
                    "both_wrong": int(((ylm != fi) & (yxg != fi) & mask).sum()),
                }
            pair_cold_complement = comp

    # ------------------ summary figures -----------------
    if macro_by_split:
        plot_generalization_drop(macro_by_split, FIGS / "fig_generalization_drop.png")
        plot_per_family_by_split(per_family_by_split, labels, FIGS / "fig_per_family_F1_by_split.png")

    if pair_cold_complement:
        plot_complementarity(
            pair_cold_complement, labels,
            FIGS / "fig_complementarity_pair_cold_llm_vs_xgb.png",
            title="Pair-cold per-family complementarity (LLM vs XGB)",
        )

    # ------------------ dataset figures -----------------
    plot_family_distribution(FIGS / "fig_dataset_family_distribution.png")
    plot_drug_degree(FIGS / "fig_dataset_drug_degree.png")
    plot_split_sizes(FIGS / "fig_dataset_split_sizes.png")

    # ------------------ reliability ---------------------
    keep_pc = _manifest(SPLITS["pair_cold"]["manifest"])
    llm_pc = _load_preds(SPLITS["pair_cold"]["llm"], keep_pc)
    confs = []
    correct = []
    for pid, rec in llm_pc.items():
        if rec["abstain"] or rec["conf"] is None or pid not in truth:
            continue
        confs.append(rec["conf"])
        correct.append(int(rec["family"] == truth[pid]))
    if confs:
        plot_reliability(confs, correct, FIGS / "fig_calibration_llm_pair_cold.png",
                         title="LLM reliability on pair_cold (greedy)")

    print(f"[done] wrote figures in {FIGS}")


if __name__ == "__main__":
    main()
