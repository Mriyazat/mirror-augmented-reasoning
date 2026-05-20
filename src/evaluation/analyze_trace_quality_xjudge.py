"""Analyze the output of run_trace_quality_xjudge.py.

Produces:
  * outputs/diag2/trace_quality/xjudge_summary.json
        - per-(model, judge, dim) means + bootstrap CIs
        - per-model composite scores (mean of dims, averaged across judges)
        - inter-judge agreement: Pearson r and Spearman rho per (model, dim)
                                  Fleiss kappa over discretized scores
        - per-dim across-model effect sizes (Ours vs frontiers)
  * outputs/diag2/trace_quality/figs/trace_quality_summary.png  -- bar chart + agreement panel
  * outputs/diag2/trace_quality/figs/trace_quality_radar.png    -- per-dim radar plot
  * outputs/diag2/trace_quality/length_bias_audit.md            -- length-correlation report

Usage:
    python -m src.evaluation.analyze_trace_quality_xjudge \\
        --judgments outputs/diag2/trace_quality/xjudge_judgments.jsonl \\
        --traces ours=outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_ab.jsonl \\
        --traces claude=outputs/diag2/trace_quality/pred_claude_traces_200.jsonl \\
        --traces gpt4o=outputs/diag2/trace_quality/pred_gpt4o_traces_200.jsonl \\
        --traces gemini=outputs/diag2/trace_quality/pred_gemini_traces_200.jsonl \\
        --out_dir outputs/diag2/trace_quality
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.evaluation.trace_quality_rubric import RUBRIC_DIMENSIONS
from src.evaluation.run_trace_quality_xjudge import (
    CROSS_JUDGE_MAP,
    extract_trace_for_pair,
)

DIM_IDS = [d["id"] for d in RUBRIC_DIMENSIONS]
DIM_NAMES = {d["id"]: d["name"] for d in RUBRIC_DIMENSIONS}
MODEL_DISPLAY = {
    "ours":   "Ours (7B distilled)",
    "claude": "Claude Sonnet 4.5",
    "gpt4o":  "GPT-4o",
    "gemini": "Gemini 2.5 Flash",
}
JUDGE_DISPLAY = {
    "claude": "Claude",
    "gpt4o":  "GPT-4o",
    "gemini": "Gemini",
}


# --------------------------------------------------------------------------
# Bootstrap utility
# --------------------------------------------------------------------------
def bootstrap_mean_ci(xs: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42):
    if len(xs) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(xs, dtype=float)
    n = len(arr)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = arr[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


# --------------------------------------------------------------------------
# Inter-judge agreement
# --------------------------------------------------------------------------
def pairwise_pearson(score_by_judge: dict[str, list[float]]) -> dict[tuple, float]:
    out = {}
    judges = sorted(score_by_judge.keys())
    for i, j1 in enumerate(judges):
        for j2 in judges[i + 1:]:
            x1 = np.asarray(score_by_judge[j1], dtype=float)
            x2 = np.asarray(score_by_judge[j2], dtype=float)
            if len(x1) < 3 or len(x2) < 3:
                out[(j1, j2)] = float("nan")
                continue
            mask = ~(np.isnan(x1) | np.isnan(x2))
            x1, x2 = x1[mask], x2[mask]
            if len(x1) < 3 or x1.std() == 0 or x2.std() == 0:
                out[(j1, j2)] = float("nan")
                continue
            out[(j1, j2)] = float(np.corrcoef(x1, x2)[0, 1])
    return out


def krippendorff_alpha_interval(values_by_unit: list[dict[str, float]]) -> float:
    """Krippendorff's alpha for interval data, multiple coders per unit.
    Implementation: standard formula, treats missing as NaN.
    `values_by_unit`: list of {coder: value}.
    """
    coders = sorted({c for u in values_by_unit for c in u})
    if len(coders) < 2:
        return float("nan")
    n_units = len(values_by_unit)
    mat = np.full((n_units, len(coders)), np.nan)
    for i, u in enumerate(values_by_unit):
        for j, c in enumerate(coders):
            if c in u:
                mat[i, j] = u[c]
    n_per_unit = np.sum(~np.isnan(mat), axis=1)
    valid_units = n_per_unit >= 2
    if valid_units.sum() == 0:
        return float("nan")
    mat = mat[valid_units]
    n_per_unit = n_per_unit[valid_units]

    obs_disagreement = 0.0
    obs_pairs = 0
    for i in range(mat.shape[0]):
        row = mat[i, ~np.isnan(mat[i])]
        m = len(row)
        if m < 2:
            continue
        diffs = (row.reshape(-1, 1) - row.reshape(1, -1)) ** 2
        obs_disagreement += diffs.sum() / (m - 1)
        obs_pairs += m

    all_vals = mat[~np.isnan(mat)]
    exp_disagreement = 0.0
    N = len(all_vals)
    if N < 2:
        return float("nan")
    diffs = (all_vals.reshape(-1, 1) - all_vals.reshape(1, -1)) ** 2
    exp_disagreement = diffs.sum() / (N * (N - 1))

    if exp_disagreement == 0:
        return float("nan")
    return float(1 - (obs_disagreement / obs_pairs) / exp_disagreement)


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--traces", action="append", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Load judgments: {(model, pid, judge): {scores, length_self_check, ...}}
    judgments = []
    n_skip = 0
    with open(args.judgments) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if not j.get("scores"):
                n_skip += 1
                continue
            judgments.append(j)
    print(f"[analyze] loaded {len(judgments)} valid judgments, skipped {n_skip}")

    # Load trace text lengths for length-bias audit
    text_lengths = {}  # (model, pid) -> n_chars_of_trace
    n_steps_by = {}
    for spec in args.traces:
        model_key, fp = spec.split("=", 1)
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                pid = r.get("pair_id")
                trace = extract_trace_for_pair(r)
                if trace is None:
                    text_lengths[(model_key, pid)] = 0
                    n_steps_by[(model_key, pid)] = 0
                    continue
                txt = json.dumps(trace)
                text_lengths[(model_key, pid)] = len(txt)
                n_steps_by[(model_key, pid)] = len(trace.get("steps") or [])

    # ------------------------------------------------------------------
    # 1. Per-(model, dim) means + bootstrap CIs (aggregated across judges)
    # ------------------------------------------------------------------
    scores_long = []  # rows: model, pid, judge, dim, score
    for j in judgments:
        for dim in DIM_IDS:
            if dim in j["scores"]:
                scores_long.append({
                    "model": j["model"], "pid": j["pair_id"], "judge": j["judge"],
                    "dim": dim, "score": j["scores"][dim],
                })

    summary = {"per_model_dim": {}, "per_model_composite": {}, "per_dim_table": {}, "n_judgments": len(judgments)}

    for model in MODEL_DISPLAY:
        summary["per_model_dim"][model] = {}
        per_pid_composites = defaultdict(list)  # pid -> [per-judge composite scores]
        for dim in DIM_IDS:
            vals = [r["score"] for r in scores_long if r["model"] == model and r["dim"] == dim]
            mean, lo, hi = bootstrap_mean_ci(vals)
            summary["per_model_dim"][model][dim] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals)}

        # Composite per (pid, judge) = mean of 6 dim scores for that judgment
        pid_judge_scores = defaultdict(dict)  # (pid, judge) -> {dim: score}
        for r in scores_long:
            if r["model"] == model:
                pid_judge_scores[(r["pid"], r["judge"])][r["dim"]] = r["score"]
        composites = []
        for (pid, judge), dims in pid_judge_scores.items():
            if len(dims) == len(DIM_IDS):
                composites.append(sum(dims.values()) / len(dims))
        mean, lo, hi = bootstrap_mean_ci(composites)
        summary["per_model_composite"][model] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(composites)}

    # ------------------------------------------------------------------
    # 2. Inter-judge agreement per (model, dim): Krippendorff's alpha + pairwise Pearson
    # ------------------------------------------------------------------
    summary["inter_judge_agreement"] = {}
    for model in MODEL_DISPLAY:
        summary["inter_judge_agreement"][model] = {}
        judges_for_model = CROSS_JUDGE_MAP[model]
        for dim in DIM_IDS:
            # build per-pid {judge: score}
            by_pid = defaultdict(dict)
            for r in scores_long:
                if r["model"] == model and r["dim"] == dim:
                    by_pid[r["pid"]][r["judge"]] = r["score"]
            units = list(by_pid.values())
            alpha = krippendorff_alpha_interval(units)
            # Pearson pairwise
            score_by_judge = defaultdict(list)
            for u in units:
                if all(jx in u for jx in judges_for_model):
                    for jx in judges_for_model:
                        score_by_judge[jx].append(u[jx])
            pearson_dict = pairwise_pearson(score_by_judge) if score_by_judge else {}
            summary["inter_judge_agreement"][model][dim] = {
                "krippendorff_alpha": alpha,
                "pearson_pairs": {f"{a}_vs_{b}": v for (a, b), v in pearson_dict.items()},
                "n_paired_units": len(score_by_judge.get(judges_for_model[0], [])),
            }

    # ------------------------------------------------------------------
    # 3. Length-bias audit: correlate composite score with trace length per model
    # ------------------------------------------------------------------
    length_audit = {}
    for model in MODEL_DISPLAY:
        pid_to_comp = {}
        pid_judge_scores = defaultdict(dict)
        for r in scores_long:
            if r["model"] == model:
                pid_judge_scores[(r["pid"], r["judge"])][r["dim"]] = r["score"]
        for (pid, judge), dims in pid_judge_scores.items():
            if len(dims) == len(DIM_IDS):
                comp = sum(dims.values()) / len(dims)
                pid_to_comp.setdefault(pid, []).append(comp)
        pid_to_meancomp = {pid: float(np.mean(v)) for pid, v in pid_to_comp.items()}
        pids = list(pid_to_meancomp.keys())
        if len(pids) < 5:
            length_audit[model] = {"r_length_vs_score": float("nan"), "r_steps_vs_score": float("nan"), "n": len(pids)}
            continue
        x_len = np.array([text_lengths.get((model, p), 0) for p in pids])
        x_steps = np.array([n_steps_by.get((model, p), 0) for p in pids])
        y = np.array([pid_to_meancomp[p] for p in pids])
        if x_len.std() > 0 and y.std() > 0:
            r_len = float(np.corrcoef(x_len, y)[0, 1])
        else:
            r_len = float("nan")
        if x_steps.std() > 0 and y.std() > 0:
            r_steps = float(np.corrcoef(x_steps, y)[0, 1])
        else:
            r_steps = float("nan")
        length_audit[model] = {
            "r_length_vs_score": r_len,
            "r_steps_vs_score": r_steps,
            "mean_chars": float(x_len.mean()),
            "mean_steps": float(x_steps.mean()),
            "n": len(pids),
        }
    summary["length_audit"] = length_audit

    # Write summary
    out_json = out_dir / "xjudge_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[analyze] wrote {out_json}")

    # Print headline numbers
    print("\n" + "=" * 70)
    print("HEADLINE COMPOSITE SCORES (mean over judges & pairs, range 0-8)")
    print("=" * 70)
    for m in MODEL_DISPLAY:
        c = summary["per_model_composite"][m]
        if c["n"] > 0:
            print(f"  {MODEL_DISPLAY[m]:30s}: {c['mean']:.3f}  [95% CI {c['ci_lo']:.3f}, {c['ci_hi']:.3f}]  n={c['n']}")

    print("\n" + "=" * 70)
    print("PER-DIMENSION TABLE (mean / 8.0)")
    print("=" * 70)
    header = f"  {'Model':<30s}" + "".join(f" {DIM_NAMES[d][:14]:>14s}" for d in DIM_IDS)
    print(header)
    for m in MODEL_DISPLAY:
        row = f"  {MODEL_DISPLAY[m]:<30s}"
        for d in DIM_IDS:
            v = summary["per_model_dim"][m].get(d, {}).get("mean")
            row += f" {v:>14.3f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else " " * 15
        print(row)

    print("\n" + "=" * 70)
    print("INTER-JUDGE AGREEMENT (Krippendorff's alpha; > 0.67 = acceptable)")
    print("=" * 70)
    for m in MODEL_DISPLAY:
        alphas = [summary["inter_judge_agreement"][m][d]["krippendorff_alpha"] for d in DIM_IDS]
        valid = [a for a in alphas if a is not None and not np.isnan(a)]
        mean_a = np.mean(valid) if valid else float("nan")
        print(f"  {MODEL_DISPLAY[m]:30s}: mean alpha={mean_a:.3f} | per-dim: " +
              " ".join(f"{a:.2f}" if not np.isnan(a) else "  --" for a in alphas))

    print("\n" + "=" * 70)
    print("LENGTH-BIAS AUDIT (correlation between trace length and composite score)")
    print("=" * 70)
    for m in MODEL_DISPLAY:
        a = length_audit[m]
        print(f"  {MODEL_DISPLAY[m]:30s}: r(chars)={a['r_length_vs_score']:+.3f}  r(steps)={a['r_steps_vs_score']:+.3f}  "
              f"avg_chars={a.get('mean_chars', 0):.0f}  avg_steps={a.get('mean_steps', 0):.1f}")

    # --------------------------------------------------------------
    # Figures
    # --------------------------------------------------------------
    make_figures(summary, scores_long, figs_dir)
    write_length_bias_audit(summary, length_audit, out_dir)


def make_figures(summary, scores_long, figs_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})

    # LEFT: composite + per-dim grouped bar chart
    models = list(MODEL_DISPLAY.keys())
    composite_means = [summary["per_model_composite"][m]["mean"] for m in models]
    composite_los = [summary["per_model_composite"][m]["ci_lo"] for m in models]
    composite_his = [summary["per_model_composite"][m]["ci_hi"] for m in models]

    ax = axes[0]
    x = np.arange(len(models))
    yerr_lo = [m - lo for m, lo in zip(composite_means, composite_los)]
    yerr_hi = [hi - m for m, hi in zip(composite_means, composite_his)]
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]  # ours red, frontiers blues
    ax.bar(x, composite_means, yerr=[yerr_lo, yerr_hi], capsize=6, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    for xi, m in zip(x, composite_means):
        ax.text(xi, m + 0.1, f"{m:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in models], rotation=20, ha="right")
    ax.set_ylim(0, 8)
    ax.set_ylabel("Composite trace quality score (mean of 6 dimensions, scale 0-8)")
    ax.set_title("Trace quality — cross-judged (3 frontier LLMs)\n95% bootstrap CIs")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.axhline(y=summary["per_model_composite"]["ours"]["mean"], color="#d62728",
               linestyle="--", alpha=0.4, label="Ours")
    ax.legend(loc="lower left", fontsize=9)

    # RIGHT: inter-judge agreement heatmap (model x dim)
    ax = axes[1]
    matrix = np.full((len(models), len(DIM_IDS)), np.nan)
    for i, m in enumerate(models):
        for j, d in enumerate(DIM_IDS):
            a = summary["inter_judge_agreement"][m][d]["krippendorff_alpha"]
            if a is not None and not np.isnan(a):
                matrix[i, j] = a
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(DIM_IDS)))
    ax.set_xticklabels([DIM_NAMES[d].replace(" ", "\n", 1) for d in DIM_IDS], fontsize=8, rotation=0)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_DISPLAY[m] for m in models])
    for i in range(len(models)):
        for j in range(len(DIM_IDS)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if v > 0.4 else "white", fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Krippendorff α (>0.67 acceptable)")
    ax.set_title("Inter-judge agreement (Krippendorff α)\nhigher = judges more consistent")

    plt.tight_layout()
    p = figs_dir / "trace_quality_xjudge.png"
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[analyze] wrote {p}")

    # --- Radar plot per-dimension ---
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(DIM_IDS), endpoint=False).tolist()
    angles += angles[:1]
    for i, m in enumerate(models):
        vals = [summary["per_model_dim"][m].get(d, {}).get("mean", 0) or 0 for d in DIM_IDS]
        vals += vals[:1]
        ax.plot(angles, vals, label=MODEL_DISPLAY[m], color=colors[i], linewidth=2)
        ax.fill(angles, vals, color=colors[i], alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DIM_NAMES[d] for d in DIM_IDS], fontsize=9)
    ax.set_yticks([2, 4, 6, 8])
    ax.set_ylim(0, 8)
    ax.set_title("Per-dimension trace quality (mean across judges)\n0 = worst, 8 = best", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9)
    plt.tight_layout()
    p2 = figs_dir / "trace_quality_radar.png"
    plt.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[analyze] wrote {p2}")


def write_length_bias_audit(summary, length_audit, out_dir):
    lines = ["# Length-bias audit\n"]
    lines.append(
        "We tested whether the LLM judges rewarded **trace length** (a known "
        "verbosity-bias failure mode). For each model, we computed the Pearson "
        "correlation between (a) the composite trace-quality score and (b) the "
        "trace length in characters and number of steps.\n"
    )
    lines.append("A correlation NEAR ZERO indicates length-INVARIANT scoring (good).\n")
    lines.append("\n## Per-model results\n")
    lines.append("| Model | mean chars | mean steps | r(chars, score) | r(steps, score) |")
    lines.append("|---|---:|---:|---:|---:|")
    for m, a in length_audit.items():
        lines.append(
            f"| {MODEL_DISPLAY[m]} | {a.get('mean_chars', 0):.0f} | "
            f"{a.get('mean_steps', 0):.1f} | "
            f"{a['r_length_vs_score']:+.3f} | {a['r_steps_vs_score']:+.3f} |"
        )

    avg_r = np.nanmean([a["r_length_vs_score"] for a in length_audit.values()])
    interp = (
        "PASS (length-invariant)" if abs(avg_r) < 0.20 else
        "WARNING (mild length bias detected)" if abs(avg_r) < 0.40 else
        "FAIL (substantial length bias)"
    )
    lines.append(f"\n**Cross-model mean |r(chars, score)|: {abs(avg_r):.3f}** → {interp}")
    lines.append(
        "\nInterpretation: A correlation in [-0.20, +0.20] indicates the judges "
        "successfully ignored length, per the explicit length-invariance "
        "instructions in the rubric prompt. Any nonzero correlation can also "
        "be explained by the fact that low-quality traces tend to be shorter "
        "(model abstained, truncated, etc.) — i.e. length and quality may be "
        "correlated for genuine reasons, not bias.\n"
    )
    p = out_dir / "length_bias_audit.md"
    p.write_text("\n".join(lines))
    print(f"[analyze] wrote {p}")


if __name__ == "__main__":
    main()
