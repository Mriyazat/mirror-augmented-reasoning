"""Comprehensive visualization suite for the paper.

Generates many figures into outputs/diag2/figs/:
  perfamily_f1_heatmap.png       per-family F1 across all models, all splits
  confusion_matrices.png         4-up: Ours vs MLP vs Claude vs GPT-4o (random_full)
  cost_vs_f1_pareto.png          log-x cost vs F1 with model labels
  reliability_diagram.png        per-bucket accuracy vs confidence
  oracle_ceiling.png             per-split: best-of-modes ceiling vs achieved
  error_overlap_venn.png         which models fail on the same cases
  cross_llm_agreement.png        agreement % matrix between LLMs
  selective_coverage_3splits.png coverage-accuracy curves
  trace_step_distribution.png    trace length distribution by correctness
  subtype_coverage.png           # subtypes correctly predicted by family
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pyarrow.parquet as pq
from sklearn.metrics import f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "outputs/diag2/figs"
FIGS.mkdir(parents=True, exist_ok=True)

FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]
FAM_SHORT = ["AdvR", "Eff", "PD", "PK-Abs", "PK-Dis", "PK-Exc", "PK-Met"]
SPLITS = ["random_full", "drug_cold", "pair_cold"]

# ---- helpers ----

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
                "subtype": fp.get("subtype"),
                "abstain": bool(fp.get("abstain", False)),
                "conf": float(fp.get("confidence") or 0.5),
                "trace_maj": _trace_majority(r.get("trace")),
                "n_steps": len((r.get("trace") or {}).get("steps") or []),
            }
    return out


def _truth():
    rows = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet",
                         columns=["pair_id", "family", "subtype"]).to_pylist()
    fam = {r["pair_id"]: r["family"] for r in rows}
    sub = {r["pair_id"]: r["subtype"] for r in rows}
    return fam, sub


def _manifest(split):
    with open(ROOT / f"outputs/eval_prompts/{split}_test_5000_stratified.manifest.jsonl") as f:
        return set(json.loads(l)["pair_id"] for l in f)


# Best LLM prediction file per split (rerank8 for random, rerank4 for cold)
LLM_PRED = {
    "random_full": ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank8_abba.jsonl",
    "drug_cold":   ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_drug_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
    "pair_cold":   ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_pair_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
}
MLP_PRED = {
    "random_full": ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_random_full_5k.jsonl",
    "drug_cold":   ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_drug_cold_5k.jsonl",
    "pair_cold":   ROOT / "outputs/baselines_perpair/preds_deepddi_mlp_fast_pair_cold_5k.jsonl",
}
XGB_PRED = {
    "random_full": ROOT / "outputs/baselines_perpair/preds_xgb_random_full.jsonl",
    "drug_cold":   ROOT / "outputs/baselines_perpair/preds_xgb_drug_cold.jsonl",
    "pair_cold":   ROOT / "outputs/baselines_perpair/preds_xgb_pair_cold.jsonl",
}
GREEDY_PRED = {
    "random_full": ROOT / "outputs/eval_prompts/pre_sft_greedy_baselines/pred_phase4_random_full_greedy.jsonl",
}
GPT4O_PRED = {
    "random_full": ROOT / "outputs/eval_prompts/pred_gpt4o_random_full_500.jsonl",
    "drug_cold":   ROOT / "outputs/eval_prompts/pred_gpt4o_drug_cold_500.jsonl",
    "pair_cold":   ROOT / "outputs/eval_prompts/pred_gpt4o_pair_cold_500.jsonl",
}
CLAUDE_PRED = {
    "random_full": ROOT / "outputs/eval_prompts/pred_claude_sonnet_random_full_500.jsonl",
    "drug_cold":   ROOT / "outputs/eval_prompts/pred_claude_sonnet_drug_cold_500.jsonl",
    "pair_cold":   ROOT / "outputs/eval_prompts/pred_claude_sonnet_pair_cold_500.jsonl",
}
MED42_PRED = {
    "random_full": ROOT / "outputs/eval_prompts/pred_med42_random_full_500.jsonl",
    "drug_cold":   ROOT / "outputs/eval_prompts/pred_med42_drug_cold_500.jsonl",
    "pair_cold":   ROOT / "outputs/eval_prompts/pred_med42_pair_cold_500.jsonl",
}

# =====================================================================
# 1. Per-family F1 heatmap across all models and splits
# =====================================================================

def fig_perfamily_f1_heatmap():
    truth_fam, _ = _truth()
    models = ["Ours-7B", "GPT-4o", "Claude-S4.6", "Med42-8B", "MLP", "XGB"]
    src_paths = {"Ours-7B": LLM_PRED, "GPT-4o": GPT4O_PRED,
                 "Claude-S4.6": CLAUDE_PRED, "Med42-8B": MED42_PRED,
                 "MLP": MLP_PRED, "XGB": XGB_PRED}
    rows = []
    row_labels = []
    for split in SPLITS:
        keep = _manifest(split)
        srcs = {m: _load(src_paths[m].get(split), keep) for m in models}
        # restrict to intersection so all bars are over the same set
        nonempty = {m: s for m, s in srcs.items() if s}
        if not nonempty:
            continue
        non_ours = [s for m, s in nonempty.items() if m != "Ours-7B"]
        anchor = min(non_ours, key=len) if non_ours else nonempty["Ours-7B"]
        common = set(anchor.keys())
        for m, s in nonempty.items():
            common &= set(s.keys())
        common = sorted(common & set(truth_fam.keys()))
        if not common:
            continue
        for m in models:
            row = []
            if m not in nonempty:
                row = [np.nan] * len(FAMS)
            else:
                yt = [truth_fam[p] for p in common]
                yp = [(nonempty[m][p]["final"] or "PD_Activity") for p in common]
                per = f1_score(yt, yp, labels=FAMS, average=None, zero_division=0)
                row = list(per)
            rows.append(row)
            row_labels.append(f"{m} | {split}")

    data = np.array(rows)
    fig, ax = plt.subplots(figsize=(8, max(6, 0.35 * len(rows))))
    cmap = LinearSegmentedColormap.from_list("greenred", ["#cc3a3a", "#f7f7f7", "#2ca02c"])
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1.0)
    ax.set_xticks(range(len(FAMS)))
    ax.set_xticklabels(FAM_SHORT, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7)
            else:
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if v < 0.3 or v > 0.85 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("per-family F1", fontsize=9)
    ax.set_title("Per-family F1: every model × every split\n"
                 "(grey lines separate splits)", fontsize=11)
    fig.tight_layout()
    out = FIGS / "perfamily_f1_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# =====================================================================
# 2. Confusion matrices: Ours vs MLP vs Claude vs GPT-4o (random_full 500)
# =====================================================================

def fig_confusion_matrices():
    truth_fam, _ = _truth()
    keep = _manifest("random_full")
    srcs = {
        "Ours (7B)":  _load(LLM_PRED["random_full"], keep),
        "Claude S4.6": _load(CLAUDE_PRED["random_full"], keep),
        "GPT-4o":      _load(GPT4O_PRED["random_full"], keep),
        "MLP (DeepDDI)": _load(MLP_PRED["random_full"], keep),
    }
    # intersection so apples-to-apples (Claude/GPT-4o only have 500)
    common = set.intersection(*[set(s.keys()) for s in srcs.values() if s])
    common = sorted(common & set(truth_fam.keys()))
    yt = [truth_fam[p] for p in common]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (name, src) in zip(axes, srcs.items()):
        yp = [(src[p]["final"] or "PD_Activity") for p in common]
        cm = confusion_matrix(yt, yp, labels=FAMS)
        cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(FAMS)))
        ax.set_xticklabels(FAM_SHORT, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(FAMS)))
        ax.set_yticklabels(FAM_SHORT, fontsize=8)
        ax.set_xlabel("predicted")
        if name == "Ours (7B)":
            ax.set_ylabel("true")
        for i in range(len(FAMS)):
            for j in range(len(FAMS)):
                v = cm_n[i, j]
                if v > 0.05:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.5 else "black")
        f1m = f1_score(yt, yp, labels=FAMS, average="macro", zero_division=0)
        ax.set_title(f"{name}\nrandom_full (n={len(common)}) F1={f1m:.3f}", fontsize=10)
    fig.suptitle("Row-normalized confusion matrices: diagonal = recall per family",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGS / "confusion_matrices.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# =====================================================================
# 3. Cost vs F1 Pareto plot
# =====================================================================

def fig_cost_vs_f1():
    # Mean F1 across 3 splits (from llm_vs_llm.json)
    llmvllm = json.loads((ROOT / "outputs/diag2/headline/llm_vs_llm.json").read_text())
    means = defaultdict(list)
    for sp in SPLITS:
        if sp not in llmvllm:
            continue
        for name, e in llmvllm[sp]["models"].items():
            means[name].append(e["macro_F1"])
    cost = {
        "Ours (7B local)":   1e-4,
        "Med42-v2-8B":        1.2e-4,
        "BioMistral-7B":      1.2e-4,
        "OpenBioLLM-8B":      1.4e-4,
        "GPT-4o (~200B)":     1.2e-2,
        "Claude Sonnet 4.6":  8e-3,
    }

    points = []
    for k, vs in means.items():
        avg = float(np.mean(vs))
        # find matching cost label
        c = None
        if k.startswith("Ours") and "stack" in k:
            c = cost["Ours (7B local)"]; lbl = "Ours-7B (stack)"
        elif k.startswith("Ours"):
            c = cost["Ours (7B local)"]; lbl = "Ours-7B"
        elif "GPT-4o" in k:
            c = cost["GPT-4o (~200B)"]; lbl = "GPT-4o (~200B)"
        elif "Claude" in k:
            c = cost["Claude Sonnet 4.6"]; lbl = "Claude Sonnet 4.6"
        elif "Med42" in k:
            c = cost["Med42-v2-8B"]; lbl = "Med42-v2-8B"
        elif "BioMistral" in k:
            c = cost["BioMistral-7B"]; lbl = "BioMistral-7B"
        elif "OpenBio" in k:
            c = cost["OpenBioLLM-8B"]; lbl = "OpenBioLLM-8B"
        if c is None:
            continue
        points.append((lbl, c, avg))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors_map = {
        "Ours-7B": "#2ca02c", "Ours-7B (stack)": "#2ca02c",
        "GPT-4o (~200B)": "#ff7f0e",
        "Claude Sonnet 4.6": "#1f77b4",
        "Med42-v2-8B": "#d62728", "BioMistral-7B": "#9467bd",
        "OpenBioLLM-8B": "#8c564b",
    }
    for name, c, f1 in points:
        ax.scatter([c], [f1], s=200, color=colors_map.get(name, "#444"),
                   edgecolor="black", zorder=5)
        # offset labels to avoid overlap
        dx, dy = 0.15, 0.015
        if "Ours" in name:
            dx, dy = 0.3, 0.005
        ax.annotate(f"{name}\n(F1={f1:.3f})",
                    xy=(c, f1), xytext=(c * (1 + dx), f1 + dy),
                    fontsize=9, ha="left",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="grey"))
    ax.set_xscale("log")
    ax.set_xlabel("USD per query (log scale)", fontsize=10)
    ax.set_ylabel("mean macro-F1 across 3 test splits", fontsize=10)
    ax.set_title("Cost vs accuracy frontier (DDI family classification)\n"
                 "Top-left dominates: high F1 at low cost", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0.0, 0.65)
    out = FIGS / "cost_vs_f1_pareto.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# =====================================================================
# 4. Reliability diagram (calibration)
# =====================================================================

def fig_reliability():
    truth_fam, _ = _truth()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, split in zip(axes, SPLITS):
        keep = _manifest(split)
        src = _load(LLM_PRED[split], keep)
        common = sorted(set(src.keys()) & set(truth_fam.keys()))
        confs = np.array([src[p]["conf"] for p in common])
        correct = np.array([1.0 if (src[p]["final"] == truth_fam[p]) else 0.0
                            for p in common])
        bins = np.linspace(0.0, 1.0, 11)
        bin_idx = np.digitize(confs, bins) - 1
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        accs = []
        sizes = []
        for b in range(len(bin_centers)):
            mask = (bin_idx == b)
            if mask.sum() > 0:
                accs.append(correct[mask].mean())
                sizes.append(mask.sum())
            else:
                accs.append(np.nan)
                sizes.append(0)
        ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
        valid = ~np.isnan(accs)
        ax.scatter(bin_centers[valid], np.array(accs)[valid],
                   s=[max(20, s / 5) for s in np.array(sizes)[valid]],
                   color="#1f77b4", alpha=0.7, edgecolor="black", label="LLM")
        # ECE
        n_total = len(correct)
        ece = 0.0
        for c, a, s in zip(bin_centers, accs, sizes):
            if s > 0 and not np.isnan(a):
                ece += abs(c - a) * s / n_total
        ax.set_xlabel("confidence")
        if split == "random_full":
            ax.set_ylabel("empirical accuracy")
        ax.set_title(f"{split}  ECE={ece:.3f}  n={n_total}", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Reliability diagram (LLM, rerank): bubble size = bin count",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIGS / "reliability_diagram.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# =====================================================================
# 5. Oracle best-of-modes ceiling
# =====================================================================

def fig_oracle_ceiling():
    truth_fam, _ = _truth()
    fig, ax = plt.subplots(figsize=(8, 5))
    splits = SPLITS
    bars_named = ["greedy", "rerank4", "rerank8", "vote3", "ORACLE best-of-modes"]
    x = np.arange(len(splits))
    width = 0.16
    colors = ["#aaaaaa", "#4292c6", "#2171b5", "#2ca02c", "#d62728"]
    data = {b: [] for b in bars_named}
    for split in splits:
        keep = _manifest(split)
        srcs = {
            "rerank8": _load(ROOT / f"outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_{split}_test_5000_stratified_nb_rerank8_abba.jsonl", keep) if split == "random_full" else _load(LLM_PRED[split], keep),
            "rerank4": _load(ROOT / f"outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_{split}_test_5000_stratified_nb_rerank4_abba.jsonl", keep),
        }
        if split == "random_full":
            srcs["greedy"] = _load(GREEDY_PRED["random_full"], keep)
        else:
            srcs["greedy"] = {}
        srcs = {k: v for k, v in srcs.items() if v}
        common_keys = [set(v.keys()) for v in srcs.values()]
        common = sorted(set.intersection(*common_keys) & set(truth_fam.keys())) if common_keys else []
        yt = [truth_fam[p] for p in common]
        per_mode = {}
        for name, src in srcs.items():
            yp = [(src[p]["final"] or "PD_Activity") for p in common]
            per_mode[name] = (yp, f1_score(yt, yp, labels=FAMS, average="macro", zero_division=0))

        # vote3 only when greedy + rerank8 exist
        vote3_yp = None
        if "greedy" in srcs and "rerank8" in srcs:
            vote3_yp = []
            for p in common:
                g = srcs["greedy"][p]; r = srcs["rerank8"][p]
                tm = r["trace_maj"] or g["trace_maj"]
                preds = [g["final"], r["final"], tm]
                wts = [g["conf"], r["conf"], 0.6]
                c = Counter()
                for prd, w in zip(preds, wts):
                    if prd in FAMS:
                        c[prd] += w
                top = c.most_common(1)[0][0] if c else r["final"]
                vote3_yp.append(top if top in FAMS else "PD_Activity")
            per_mode["vote3"] = (vote3_yp, f1_score(yt, vote3_yp, labels=FAMS, average="macro", zero_division=0))

        # ORACLE: per-pair, best of any mode
        oracle_yp = []
        for i, p in enumerate(common):
            opts = set()
            for name, (yp_list, _) in per_mode.items():
                opts.add(yp_list[i])
            if yt[i] in opts:
                oracle_yp.append(yt[i])
            else:
                # fallback: pick the most confident rerank8
                oracle_yp.append(per_mode.get("rerank8", per_mode.get("rerank4"))[0][i])
        oracle_f1 = f1_score(yt, oracle_yp, labels=FAMS, average="macro", zero_division=0)

        data["greedy"].append(per_mode.get("greedy", (None, 0.0))[1])
        data["rerank4"].append(per_mode.get("rerank4", (None, 0.0))[1])
        data["rerank8"].append(per_mode.get("rerank8", (None, 0.0))[1])
        data["vote3"].append(per_mode.get("vote3", (None, 0.0))[1])
        data["ORACLE best-of-modes"].append(oracle_f1)

    for i, b in enumerate(bars_named):
        offset = (i - 2) * width
        ax.bar(x + offset, data[b], width, color=colors[i], label=b)
        for j, v in enumerate(data[b]):
            if v > 0:
                ax.text(x[j] + offset, v + 0.005, f"{v:.2f}",
                        ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=10)
    ax.set_ylabel("macro-F1")
    ax.set_title("Aggregation ceiling: F1 by mode + oracle 'pick the right one'\n"
                 "Gap between vote3 and oracle = how much smarter aggregation is still possible",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    out = FIGS / "oracle_ceiling.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# =====================================================================
# 6. Error overlap (which models fail together) — random_full 500
# =====================================================================

def fig_error_overlap():
    truth_fam, _ = _truth()
    keep = _manifest("random_full")
    srcs = {
        "Ours (7B)":  _load(LLM_PRED["random_full"], keep),
        "Claude S4.6": _load(CLAUDE_PRED["random_full"], keep),
        "GPT-4o":      _load(GPT4O_PRED["random_full"], keep),
    }
    common = set.intersection(*[set(s.keys()) for s in srcs.values() if s])
    common = sorted(common & set(truth_fam.keys()))
    err = {n: {p for p in common if (s.get(p, {}).get("final") != truth_fam[p])}
           for n, s in srcs.items()}

    # Pairwise overlap counts
    names = list(srcs.keys())
    overlap = np.zeros((len(names), len(names)), dtype=int)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            overlap[i, j] = len(err[ni] & err[nj])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    im = ax.imshow(overlap, cmap="Reds")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(overlap[i, j]),
                    ha="center", va="center", fontsize=10,
                    color="white" if overlap[i, j] > overlap.max() * 0.6 else "black")
    ax.set_title(f"# pairs both models get WRONG (random_full, n={len(common)})", fontsize=10)

    # Bar chart: unique errors per model
    ax = axes[1]
    only_ours = err["Ours (7B)"] - err["Claude S4.6"] - err["GPT-4o"]
    only_claude = err["Claude S4.6"] - err["Ours (7B)"] - err["GPT-4o"]
    only_gpt = err["GPT-4o"] - err["Ours (7B)"] - err["Claude S4.6"]
    all_wrong = err["Ours (7B)"] & err["Claude S4.6"] & err["GPT-4o"]
    cats = ["Only Ours\nwrong", "Only Claude\nwrong", "Only GPT-4o\nwrong",
            "All 3 wrong\n(hard cases)"]
    vals = [len(only_ours), len(only_claude), len(only_gpt), len(all_wrong)]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    bars = ax.bar(cats, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, str(v),
                ha="center", fontsize=10)
    ax.set_ylabel("# pairs")
    ax.set_title(f"Unique vs shared errors (n={len(common)})\n"
                 f"All 3 wrong = inherently hard ({len(all_wrong)}/{len(common)} = "
                 f"{len(all_wrong)/len(common):.0%})", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "error_overlap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# =====================================================================
# 7. Cross-LLM agreement matrix
# =====================================================================

def fig_cross_llm_agreement():
    truth_fam, _ = _truth()
    keep = _manifest("random_full")
    srcs = {
        "Ours (7B)":  _load(LLM_PRED["random_full"], keep),
        "Claude S4.6": _load(CLAUDE_PRED["random_full"], keep),
        "GPT-4o":      _load(GPT4O_PRED["random_full"], keep),
        "Med42-8B":    _load(MED42_PRED["random_full"], keep),
    }
    common = set.intersection(*[set(s.keys()) for s in srcs.values() if s])
    common = sorted(common & set(truth_fam.keys()))
    names = list(srcs.keys())
    n = len(names)
    agree = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            agree[i, j] = np.mean([
                srcs[ni][p]["final"] == srcs[nj][p]["final"] for p in common
            ])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(agree, cmap="Blues", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{agree[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if agree[i, j] > 0.7 else "black", fontsize=10)
    ax.set_title(f"Cross-LLM agreement on family predictions\n"
                 f"random_full (n={len(common)})", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7).set_label("agreement", fontsize=9)
    fig.tight_layout()
    out = FIGS / "cross_llm_agreement.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# =====================================================================
# 8. Selective coverage-accuracy curves (overlay 3 splits)
# =====================================================================

def fig_selective_curves():
    truth_fam, _ = _truth()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = {"random_full": "#2ca02c", "drug_cold": "#1f77b4", "pair_cold": "#d62728"}
    for split in SPLITS:
        keep = _manifest(split)
        src = _load(LLM_PRED[split], keep)
        common = sorted(set(src.keys()) & set(truth_fam.keys()))
        confs = np.array([src[p]["conf"] for p in common])
        correct = np.array([1.0 if (src[p]["final"] == truth_fam[p]) else 0.0
                            for p in common])
        order = np.argsort(-confs)
        confs_s = confs[order]; correct_s = correct[order]
        cum_correct = np.cumsum(correct_s)
        coverage = np.arange(1, len(confs_s) + 1) / len(confs_s)
        acc = cum_correct / np.arange(1, len(confs_s) + 1)
        ax.plot(coverage, acc, color=cmap[split], lw=2, label=split)
    ax.set_xlabel("coverage (fraction of test predictions kept, highest confidence first)")
    ax.set_ylabel("accuracy among kept predictions")
    ax.set_title("Coverage–accuracy trade-off\n"
                 "Right edge = answer everything; left edge = answer only the most confident", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    out = FIGS / "selective_coverage_3splits.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# =====================================================================
# 9. Trace step distribution: how many reasoning steps per pair, by correctness
# =====================================================================

def fig_trace_step_distribution():
    truth_fam, _ = _truth()
    keep = _manifest("random_full")
    src = _load(LLM_PRED["random_full"], keep)
    common = sorted(set(src.keys()) & set(truth_fam.keys()))

    steps_correct = []
    steps_wrong = []
    for p in common:
        s = src[p]["n_steps"]
        if src[p]["final"] == truth_fam[p]:
            steps_correct.append(s)
        else:
            steps_wrong.append(s)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    bins = np.arange(0, max(max(steps_correct), max(steps_wrong)) + 2) - 0.5
    ax.hist([steps_correct, steps_wrong], bins=bins,
            label=[f"correct (n={len(steps_correct)})",
                   f"wrong (n={len(steps_wrong)})"],
            color=["#2ca02c", "#d62728"], alpha=0.7)
    ax.set_xlabel("# reasoning steps")
    ax.set_ylabel("# pairs")
    ax.set_title(f"Trace step distribution by outcome (random_full)\n"
                 f"Correct mean={np.mean(steps_correct):.2f}, "
                 f"Wrong mean={np.mean(steps_wrong):.2f}", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Per-family trace step
    ax = axes[1]
    by_fam = defaultdict(list)
    for p in common:
        by_fam[truth_fam[p]].append(src[p]["n_steps"])
    means = [np.mean(by_fam[f]) for f in FAMS]
    ax.bar(FAM_SHORT, means, color="#4292c6")
    for i, m in enumerate(means):
        ax.text(i, m + 0.05, f"{m:.2f}", ha="center", fontsize=8)
    ax.set_xticklabels(FAM_SHORT, rotation=0, fontsize=8)
    ax.set_ylabel("mean # reasoning steps")
    ax.set_title("Mean trace length by gold family", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "trace_step_distribution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# =====================================================================
# 10. Subtype coverage: # unique subtypes correctly predicted per family
# =====================================================================

def fig_subtype_coverage():
    truth_fam, truth_sub = _truth()
    keep = _manifest("random_full")
    src = _load(LLM_PRED["random_full"], keep)
    common = sorted(set(src.keys()) & set(truth_fam.keys()))

    correct_subs = defaultdict(set)
    all_subs = defaultdict(set)
    for p in common:
        gf = truth_fam[p]
        gs = truth_sub.get(p)
        all_subs[gf].add(gs)
        if (not src[p]["abstain"]
                and src[p]["final"] == gf
                and src[p]["subtype"] == gs):
            correct_subs[gf].add(gs)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(FAMS))
    correct_counts = [len(correct_subs[f]) for f in FAMS]
    all_counts = [len(all_subs[f]) for f in FAMS]
    ax.bar(x, all_counts, color="#cccccc", label="# subtypes in family (truth)")
    ax.bar(x, correct_counts, color="#2ca02c",
           label="# subtypes the LLM ever predicts correctly")
    for i, (c, a) in enumerate(zip(correct_counts, all_counts)):
        ax.text(i, max(c, a) + 0.4, f"{c}/{a}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(FAM_SHORT, fontsize=9)
    ax.set_ylabel("# unique subtypes")
    ax.set_title("Subtype prediction coverage by family (random_full)\n"
                 "Baselines: 0/all because they cannot output a subtype",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "subtype_coverage.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    print("=== generating comprehensive figures ===")
    fig_perfamily_f1_heatmap()
    fig_confusion_matrices()
    fig_cost_vs_f1()
    fig_reliability()
    fig_oracle_ceiling()
    fig_error_overlap()
    fig_cross_llm_agreement()
    fig_selective_curves()
    fig_trace_step_distribution()
    fig_subtype_coverage()
    print("=== done ===")


if __name__ == "__main__":
    main()
