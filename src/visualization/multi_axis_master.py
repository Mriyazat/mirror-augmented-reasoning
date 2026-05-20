"""Master multi-axis comparison figure for the paper.

Five panels:
  A. Family macro-F1 across splits (LLM vs LLM_stack vs MLP vs XGB)
  B. Joint family+subtype accuracy (=subtype_acc unconditional)
  C. Pair_cold rare-family F1 (where LLM dominates)
  D. Selective accuracy curve (LLM_stack with conformal thresholds)
  E. Mirror Family Stability (LLM only; baselines = N/A)

This is the figure that *replaces* a F1-only comparison and tells the
multi-axis story: "we trade some warm-condition fitting for hierarchical
prediction, cold-split generalization, structural symmetry, and
calibrated selective accuracy."
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HEAD = ROOT / "outputs/diag2/headline"
OUT = ROOT / "outputs/diag2/figs/master_multi_axis.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

SPLITS = ["random_full", "drug_cold", "pair_cold"]
SPLIT_LABELS = ["random_full", "drug_cold", "pair_cold"]
MODELS = ["LLM", "LLM_stack", "MLP", "XGB"]
COLORS = {"LLM": "#1f77b4", "LLM_stack": "#2ca02c",
          "MLP":  "#d62728", "XGB":  "#9467bd"}


def _load_split(split):
    return json.loads((HEAD / f"{split}.json").read_text())


def _load_mfs():
    return json.loads((HEAD / "mfs_summary.json").read_text())


def _load_ths():
    return json.loads((HEAD / "ths_summary.json").read_text())


def main():
    splits = {sp: _load_split(sp) for sp in SPLITS}
    mfs = _load_mfs()
    ths = _load_ths()

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    width = 0.18
    x = np.arange(len(SPLITS))

    # ---- Panel A: family macro-F1 ----
    ax = axes[0]
    for i, m in enumerate(MODELS):
        vals = [splits[sp]["models"].get(m, {}).get("macro_F1", 0) for sp in SPLITS]
        ax.bar(x + (i - 1.5) * width, vals, width, label=m, color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_LABELS, fontsize=9)
    ax.set_title("A. Family macro-F1\n(warm split: MLP wins; cold: LLM holds)", fontsize=10)
    ax.set_ylabel("macro-F1")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 1.0)

    # ---- Panel B: joint family+subtype acc (subtype_acc unconditional) ----
    ax = axes[1]
    for i, m in enumerate(MODELS):
        if m in ("LLM", "LLM_stack"):
            vals = [splits[sp]["models"].get(m, {}).get("subtype", {}).get("subtype_accuracy", 0) for sp in SPLITS]
        else:
            vals = [0 for _ in SPLITS]
        ax.bar(x + (i - 1.5) * width, vals, width, label=m, color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_LABELS, fontsize=9)
    ax.set_title("B. Family+Subtype joint accuracy\n(baselines structurally 0)", fontsize=10)
    ax.set_ylabel("exact-match acc")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.65)

    # ---- Panel C: pair_cold rare-family F1 ----
    ax = axes[2]
    for i, m in enumerate(MODELS):
        vals = [splits[sp]["models"].get(m, {}).get("rare_macro_F1", 0) for sp in SPLITS]
        ax.bar(x + (i - 1.5) * width, vals, width, label=m, color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_LABELS, fontsize=9)
    ax.set_title("C. Rare-family macro-F1\n(LLM dominates on cold splits)", fontsize=10)
    ax.set_ylabel("rare macro-F1")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    # ---- Panel D: selective accuracy curves (LLM_stack only) ----
    ax = axes[3]
    for sp in SPLITS:
        sel = splits[sp]["models"]["LLM_stack"]["selective"]
        thresholds = sorted([float(k.split("@")[-1]) for k in sel.keys()])
        accs = []
        covs = []
        for thr in thresholds:
            key = f"acc@{thr:.2f}"
            entry = sel[key]
            if entry["accuracy"] is not None:
                accs.append(entry["accuracy"])
                covs.append(entry["coverage"])
            else:
                accs.append(None)
                covs.append(None)
        pts = [(c, a) for c, a in zip(covs, accs) if a is not None]
        if pts:
            cs, as_ = zip(*sorted(pts))
            ax.plot(cs, as_, marker="o", label=sp)
    ax.set_xlabel("coverage")
    ax.set_ylabel("accuracy")
    ax.set_title("D. Selective accuracy (LLM_stack)\nconformal-thresholded abstention", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)

    # ---- Panel E: Mirror Family Stability ----
    ax = axes[4]
    vals = [mfs.get(sp, {}).get("MFS_family", 0) for sp in SPLITS]
    bars = ax.bar(x, vals, color=COLORS["LLM"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_LABELS, fontsize=9)
    ax.set_title("E. Mirror Family Stability\n(LLM only; baselines have no AB/BA notion)", fontsize=10)
    ax.set_ylabel("MFS")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Multi-axis comparison: where the distilled-7B LLM contributes beyond family F1",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
