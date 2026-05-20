"""Additional insight figures:
  pipeline_waterfall.png        Cumulative F1 gain from each pipeline stage
  family_freq_vs_f1.png         Per-family training freq vs test F1 (head/tail problem)
  cost_per_f1_point.png         How much $ to buy 1 F1 point (efficiency)
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "outputs/diag2/figs"
FIGS.mkdir(parents=True, exist_ok=True)

FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]
FAM_SHORT = ["AdvR", "Eff", "PD", "PK-Abs", "PK-Dis", "PK-Exc", "PK-Met"]


def _trace_majority(trace):
    if not isinstance(trace, dict):
        return None
    hints = [s.get("family_hint") for s in (trace.get("steps") or [])
             if s.get("family_hint") in FAMS]
    return Counter(hints).most_common(1)[0][0] if hints else None


def _load(path, keep=None):
    out = {}
    if not Path(path).exists():
        return out
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("input_order") or "ab") != "ab":
                continue
            pid = r["pair_id"]
            if keep is not None and pid not in keep:
                continue
            if pid in out:
                continue
            fp = r.get("final_prediction") or {}
            out[pid] = {
                "final": fp.get("family") if fp.get("family") in FAMS else None,
                "abstain": bool(fp.get("abstain", False)),
                "conf": float(fp.get("confidence") or 0.5),
                "trace_maj": _trace_majority(r.get("trace")),
            }
    return out


def _truth():
    rows = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet",
                         columns=["pair_id", "family"]).to_pylist()
    return {r["pair_id"]: r["family"] for r in rows}


def _manifest(split):
    with open(ROOT / f"outputs/eval_prompts/{split}_test_5000_stratified.manifest.jsonl") as f:
        return set(json.loads(l)["pair_id"] for l in f)


# =====================================================================
# pipeline_waterfall: cumulative F1 gain per stage on random_full
# =====================================================================

def fig_pipeline_waterfall():
    truth_fam = _truth()
    keep = _manifest("random_full")
    greedy = _load(ROOT / "outputs/eval_prompts/pre_sft_greedy_baselines/pred_phase4_random_full_greedy.jsonl", keep)
    rerank4 = _load(ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank4_abba.jsonl", keep)
    rerank8 = _load(ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank8_abba.jsonl", keep)

    common = sorted(set(greedy) & set(rerank4) & set(rerank8) & set(truth_fam))
    yt = [truth_fam[p] for p in common]

    def f1_of(yp):
        return f1_score(yt, yp, labels=FAMS, average="macro", zero_division=0)

    yp_g = [(greedy[p]["final"] or "PD_Activity") for p in common]
    yp_r4 = [(rerank4[p]["final"] or "PD_Activity") for p in common]
    yp_r8 = [(rerank8[p]["final"] or "PD_Activity") for p in common]

    yp_v3 = []
    for p in common:
        g = greedy[p]; r = rerank8[p]
        tm = r["trace_maj"] or g["trace_maj"]
        c = Counter()
        for pr, w in zip([g["final"], r["final"], tm], [g["conf"], r["conf"], 0.6]):
            if pr in FAMS:
                c[pr] += w
        v = c.most_common(1)[0][0] if c else r["final"]
        yp_v3.append(v if v in FAMS else "PD_Activity")

    # selective @ 0.7 coverage
    confs = np.array([rerank8[p]["conf"] for p in common])
    correct_r8 = np.array([1.0 if rerank8[p]["final"] == yt[i] else 0.0 for i, p in enumerate(common)])
    order = np.argsort(-confs)
    keep_n = int(0.70 * len(common))
    kept_idx = set(order[:keep_n].tolist())
    yp_sel = [yp_r8[i] if i in kept_idx else None for i in range(len(common))]
    # for selective, eval only on kept
    sel_yt = [yt[i] for i, p in enumerate(common) if i in kept_idx]
    sel_yp = [yp_r8[i] for i, p in enumerate(common) if i in kept_idx]
    f1_sel = f1_score(sel_yt, sel_yp, labels=FAMS, average="macro", zero_division=0)

    stages = ["greedy", "+ rerank4 ABBA", "+ rerank8 ABBA",
              "+ vote3 (CPU)", "+ selective @70% cov"]
    vals = [f1_of(yp_g), f1_of(yp_r4), f1_of(yp_r8), f1_of(yp_v3), f1_sel]
    deltas = [0] + [vals[i] - vals[i-1] for i in range(1, len(vals))]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#888888", "#4292c6", "#2171b5", "#2ca02c", "#6baed6"]
    bars = ax.bar(stages, vals, color=colors)
    for i, (b, v, d) in enumerate(zip(bars, vals, deltas)):
        if i == 0:
            ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=10)
        else:
            sign = "+" if d >= 0 else ""
            ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                    f"{v:.3f}\n({sign}{d:.3f})",
                    ha="center", fontsize=9,
                    color="green" if d > 0 else "red")
    ax.set_ylabel("macro-F1 (random_full, n=5000)")
    ax.set_title("Pipeline gain waterfall: where does each stage of the stack add F1?",
                 fontsize=11)
    ax.set_ylim(0.4, 0.7)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=10, ha="right", fontsize=9)
    fig.tight_layout()
    out = FIGS / "pipeline_waterfall.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# =====================================================================
# family_freq_vs_f1: training freq vs test F1 (head/tail problem)
# =====================================================================

def fig_family_freq_vs_f1():
    # full training data distribution
    full = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet",
                         columns=["family"]).to_pylist()
    train_dist = Counter(r["family"] for r in full)
    total = sum(train_dist.values())

    truth_fam = _truth()
    keep = _manifest("random_full")
    src = _load(ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank8_abba.jsonl", keep)
    common = sorted(set(src.keys()) & set(truth_fam))
    yt = [truth_fam[p] for p in common]
    yp = [(src[p]["final"] or "PD_Activity") for p in common]
    per_f1 = f1_score(yt, yp, labels=FAMS, average=None, zero_division=0)

    freqs = np.array([train_dist.get(f, 0) / total for f in FAMS]) * 100
    pred_dist_test = Counter(yp)
    truth_dist_test = Counter(yt)
    pred_share = np.array([pred_dist_test.get(f, 0) / len(yp) for f in FAMS]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    sc = ax.scatter(freqs, per_f1, c=range(len(FAMS)), cmap="tab10", s=200, edgecolor="black", zorder=5)
    for i, f in enumerate(FAMS):
        ax.annotate(f"{f}\n({freqs[i]:.1f}% train, F1={per_f1[i]:.2f})",
                    xy=(freqs[i], per_f1[i]),
                    xytext=(5, 8), textcoords="offset points",
                    fontsize=8)
    ax.set_xlabel("training-set prevalence (%)")
    ax.set_ylabel("test per-family F1 (random_full)")
    ax.set_title("Training prevalence vs test F1\n"
                 "AdverseRisk dominates training; rare families underfit", fontsize=10)
    ax.grid(alpha=0.3)

    # right panel: train prevalence vs pred prevalence (over-prediction bias)
    ax = axes[1]
    width = 0.35
    x = np.arange(len(FAMS))
    ax.bar(x - width/2, freqs, width, color="#aaaaaa", label="train prevalence (%)")
    ax.bar(x + width/2, pred_share, width, color="#1f77b4",
           label="test prediction share (%)")
    # truth on test = uniform 1/7 ~ 14.3%
    ax.axhline(100 / 7, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="balanced truth (14.3%)")
    ax.set_xticks(x)
    ax.set_xticklabels(FAM_SHORT, fontsize=9)
    ax.set_ylabel("% share")
    ax.set_title("Why F1 is hard: training is heavily skewed toward AdverseRisk,\n"
                 "model carries that bias into a balanced test", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "family_freq_vs_f1.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    print("=== extra insight figures ===")
    fig_pipeline_waterfall()
    fig_family_freq_vs_f1()
    print("=== done ===")


if __name__ == "__main__":
    main()
