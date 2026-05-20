"""Plot rerank4 vs greedy disagreement on val5k.

Produces two PNGs:
  outputs/diag2/figs/mode_diff_agreement.png   # stacked bar: both / rerank-only / greedy-only / both-wrong
  outputs/diag2/figs/mode_diff_rescues.png     # Venn-like bar: rerank4 vs greedy rescue overlap
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]
FAM_SHORT = ["AdvR", "Eff", "PD", "PK-Abs", "PK-Dis", "PK-Exc", "PK-Met"]


def main():
    src = ROOT / "outputs/student/trace_align/rescue_data/mode_diff.json"
    obj = json.loads(src.read_text())
    out_dir = ROOT / "outputs/diag2/figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: per-family stacked bar of disagreement ----
    n_fam = len(FAMS)
    rerank_only = np.array([obj["per_family"].get(f, {}).get("rerank_only", 0) for f in FAMS])
    greedy_only = np.array([obj["per_family"].get(f, {}).get("greedy_only", 0) for f in FAMS])
    both_wrong  = np.array([obj["per_family"].get(f, {}).get("both_wrong", 0)  for f in FAMS])
    totals = rerank_only + greedy_only + both_wrong

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(n_fam)
    ax.bar(x, both_wrong, color="#b30000", label="both wrong")
    ax.bar(x, rerank_only, bottom=both_wrong, color="#2b8cbe",
           label="rerank4 only correct")
    ax.bar(x, greedy_only, bottom=both_wrong + rerank_only, color="#74c476",
           label="greedy only correct")
    ax.set_xticks(x)
    ax.set_xticklabels(FAM_SHORT, rotation=0, fontsize=9)
    ax.set_ylabel("# val5k pairs")
    ax.set_title("Where greedy and rerank4 disagree (val5k, n=5000)")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.grid(axis="y", alpha=0.25)
    for i, tot in enumerate(totals):
        ax.text(x[i], tot + 8, f"{tot}", ha="center", fontsize=8)
    fig.tight_layout()
    p1 = out_dir / "mode_diff_agreement.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: rescue overlap (rerank4 vs greedy) ----
    only_r = obj["rerank_unique_rescues"]
    only_g = obj["greedy_unique_rescues"]
    both   = obj["both_rescues"]
    union  = obj["union_rescues"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    cats = ["rerank4 only", "both", "greedy only"]
    vals = [only_r, both, only_g]
    colors = ["#2b8cbe", "#a6611a", "#74c476"]
    bars = ax.bar(cats, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3,
                f"{v}", ha="center", fontsize=10)
    ax.set_ylabel("# rescue candidates")
    ax.set_title(f"Trace-align rescue overlap (union = {union}, +{only_g} from greedy)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p2 = out_dir / "mode_diff_rescues.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
