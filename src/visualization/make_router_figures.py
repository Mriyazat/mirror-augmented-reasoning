"""Visualize confusion-aware router gains and post-router confusion matrices."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

ROUTER_DIR = ROOT / "outputs" / "diag2" / "router"

SPLITS = ["random_full", "drug_cold", "pair_cold"]
BASES = ["xgb", "mlp"]


def gain_bar():
    data = {}
    for sp in SPLITS:
        for base in BASES:
            p = ROUTER_DIR / f"{sp}_{base}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            data[(sp, base)] = r

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SPLITS))
    width = 0.18
    llm_solo = [data[(sp, "xgb")]["test_macro_llm_solo"] for sp in SPLITS]
    xgb_solo = [data[(sp, "xgb")]["test_macro_base_solo"] for sp in SPLITS]
    mlp_solo = [data[(sp, "mlp")]["test_macro_base_solo"] for sp in SPLITS]
    router_xgb = [data[(sp, "xgb")]["test_macro_router"] for sp in SPLITS]
    router_mlp = [data[(sp, "mlp")]["test_macro_router"] for sp in SPLITS]

    bars = [
        (llm_solo, "LLM (solo)", "#d62728", -2),
        (xgb_solo, "XGB (solo)", "#2ca02c", -1),
        (mlp_solo, "MLP (solo)", "#1f77b4", 0),
        (router_xgb, "Router (LLM+XGB)", "#9467bd", 1),
        (router_mlp, "Router (LLM+MLP)", "#8c564b", 2),
    ]
    for vals, label, color, off in bars:
        positions = x + off * width
        bs = ax.bar(positions, vals, width, label=label, color=color)
        for b, v in zip(bs, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                    ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(SPLITS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Held-out test macro-F1")
    ax.set_title("Confusion-aware router gains (val-tuned, honest test)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "fig_router_gains.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def confusion_router(sp="pair_cold", base="xgb"):
    """Re-run the router policy on the same data to draw a confusion."""
    import sys
    sys.path.insert(0, str(ROOT))
    from src.evaluation.confusion_aware_router import (
        _arr, _llm_conf, _manifest, _preds, _truth, _learn_rules,
    )
    manifest = ROOT / f"outputs/eval_prompts/{sp}_test_5000_stratified.manifest.jsonl"
    llm_p = ROOT / f"outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_{sp}_test_5000_stratified_nb_abba.jsonl"
    base_p = {
        "xgb": ROOT / f"outputs/baselines_perpair/preds_xgb_{sp}.jsonl",
        "mlp": ROOT / f"outputs/baselines_perpair/preds_deepddi_mlp_fast_{sp}_5k.jsonl",
    }[base]
    if not (manifest.exists() and llm_p.exists() and base_p.exists()):
        return None
    keep = set(_manifest(manifest))
    truth = _truth(ROOT / "data_processed/labels_hierarchical.parquet", keep)
    llm = _preds(llm_p, keep)
    bp = _preds(base_p, keep)
    common = sorted(set(truth) & set(llm) & set(bp))
    rng = np.random.default_rng(42)
    idx = np.arange(len(common)); rng.shuffle(idx)
    n_val = len(common) // 2
    val_ids = [common[i] for i in idx[:n_val]]
    tst_ids = [common[i] for i in idx[n_val:]]
    labels = sorted(set(truth.values()))
    lab2idx = {f: i for i, f in enumerate(labels)}

    yv, yv_llm = _arr(val_ids, truth, llm, lab2idx)
    _, yv_base = _arr(val_ids, truth, bp, lab2idx)
    cv = _llm_conf(val_ids, llm)
    grid = [0.0, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    best_tau, best_rules, best_f1 = None, set(), -1.0
    for tau in grid:
        rules = _learn_rules(yv, yv_llm, yv_base, cv, tau, len(labels))
        mask = np.isin(yv_llm, list(rules)) & (cv < tau)
        yp = np.where(mask, yv_base, yv_llm)
        f = f1_score(yv, yp, labels=list(range(len(labels))), average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_tau, best_rules = f, tau, rules

    yt, yt_llm = _arr(tst_ids, truth, llm, lab2idx)
    _, yt_base = _arr(tst_ids, truth, bp, lab2idx)
    ct = _llm_conf(tst_ids, llm)
    apply_mask = np.isin(yt_llm, list(best_rules)) & (ct < best_tau)
    yp_router = np.where(apply_mask, yt_base, yt_llm)

    full = labels + ["__abstain__"]
    cm = confusion_matrix(yt, yp_router, labels=list(range(len(full))))
    row = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(full))); ax.set_yticks(range(len(full)))
    ax.set_xticklabels(full, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(full, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
    ax.set_title(f"Confusion - Router(LLM+{base.upper()}) on {sp}")
    for i in range(len(full)):
        for j in range(len(full)):
            v = cmn[i, j]
            if v > 0:
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = FIGS / f"fig_confusion_{sp}_router_{base}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    p = gain_bar()
    print(f"[fig] {p}")
    for sp in SPLITS:
        for b in BASES:
            o = confusion_router(sp, b)
            if o:
                print(f"[fig] {o}")


if __name__ == "__main__":
    main()
