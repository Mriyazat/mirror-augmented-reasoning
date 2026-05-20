"""Train a lightweight CPU verifier/router from val and apply to test candidates.

This trains at the *candidate* level:
  input  = features from one prediction candidate
  label  = whether candidate.family == gold.family

At test time, for each pair, score every available candidate and choose the
candidate with highest P(correct). This uses no test labels for fitting.

The val pool is narrower than the rich test pool (mainly greedy/rerank4), so
this is a conservative first learned router.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.student_ceiling_inventory import (
    FAMS, ROOT, OUT, candidate_files, final_family, load_preds, macro, truth_for_split
)
from src.evaluation.verifier_random_full_probe import verifier_flags, trace_majority, final_conf

VAL_MANIFEST = ROOT / "outputs/eval_prompts/random_full_val_5000_stratified.manifest.jsonl"
VAL_FILES = {
    "val_greedy": ROOT / "outputs/student/trace_align/rescue_data/preds_val5k_greedy.jsonl",
    "val_rerank4": ROOT / "outputs/student/trace_align/rescue_data/preds_val5k_rerank4.jsonl",
}
RICH_VAL_DIR = ROOT / "outputs/diag2/rich_val_candidates"


def load_manifest_truth(path: Path) -> dict[str, str]:
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["pair_id"]] = r["family"]
    return out


def load_records(path: Path, keep: set[str]) -> dict[str, dict]:
    out = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r.get("pair_id")
            if pid in keep and pid not in out:
                out[pid] = r
    return out


def source_features(source: str) -> list[float]:
    s = source.lower()
    return [
        float("cpu_stack" in s),
        float("rerank8" in s),
        float("rerank4" in s),
        float("greedy" in s or "pre_sft" in s),
        float("conclusion_text" in s),
        float("hint_majority" in s),
        float("hybrid" in s),
        float("verifier" in s),
        float("abba" in s),
        float("_ba" in s or s.endswith("ba")),
        float("no_neighbors" in s),
    ]


def record_features(rec: dict, source: str) -> list[float]:
    flags = verifier_flags(rec)
    fam = final_family(rec)
    tfam, tstrength = trace_majority(rec)
    trace_steps = (rec.get("trace") or {}).get("steps") or []
    conf = final_conf(rec)
    # PRM features if present.
    rr = rec.get("rerank") or {}
    chosen_prm = float(rr.get("chosen_prm") or 0.0)
    n_candidates = float(rr.get("n_candidates") or 1.0)
    feats = [
        conf,
        float(flags["violation_score"]),
        float(flags["has_gap"]),
        float(flags["has_neighbor"]),
        float(flags["speculative_conclusion"]),
        float(flags["gap_non_abstain"]),
        float(flags["weak_gap_non_abstain"]),
        float(flags["pk_metabolism_without_paired_cyp"]),
        float(flags["pk_nonmetab_without_transport"]),
        float(flags["adverse_from_gap"]),
        float(flags["invented_neighbor"]),
        float(flags["low_conf_non_abstain"]),
        float(tstrength),
        float(tfam == fam),
        float(len(trace_steps)),
        chosen_prm,
        n_candidates,
    ]
    feats.extend(float(fam == f) for f in FAMS)
    feats.extend(float(tfam == f) for f in FAMS)
    feats.extend(source_features(source))
    return feats


def build_val_dataset():
    truth = load_manifest_truth(VAL_MANIFEST)
    rows, labels = [], []
    val_files = dict(VAL_FILES)
    if RICH_VAL_DIR.exists():
        for path in sorted(RICH_VAL_DIR.glob("*.jsonl")):
            val_files[f"rich_val/{path.stem}"] = path
    for source, path in val_files.items():
        recs = load_records(path, set(truth))
        for pid, rec in recs.items():
            fam = final_family(rec)
            if fam is None:
                continue
            rows.append(record_features(rec, source))
            labels.append(int(fam == truth[pid]))
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), sorted(val_files)


def candidate_records_for_split(split: str):
    truth = truth_for_split(split)
    keep = set(truth)
    candidates = defaultdict(list)
    for path in candidate_files(split):
        # Keep only complete-ish candidate files.
        pred = load_preds(path, keep)
        if len(pred) < 4500:
            continue
        recs = load_records(path, keep)
        for pid, rec in recs.items():
            if pid in keep and final_family(rec) is not None:
                candidates[pid].append((str(path.relative_to(ROOT)), rec))
    return truth, candidates


def prepare_split(split: str):
    truth, candidates = candidate_records_for_split(split)
    pids = sorted(pid for pid, cs in candidates.items() if cs)
    rows = []
    for pid in pids:
        for source, rec in candidates[pid]:
            rows.append({
                "pid": pid,
                "features": record_features(rec, source),
                "source": source,
                "family": final_family(rec) or "PD_Activity",
            })
    X = np.asarray([r["features"] for r in rows], dtype=float)
    return truth, pids, rows, X


def eval_router(model, split_data):
    truth, pids, rows, X = split_data
    probs = model.predict_proba(X)[:, 1]
    grouped = defaultdict(list)
    for row, prob in zip(rows, probs):
        score = float(prob)
        # tiny deterministic tie-breaker toward current best-style modes
        if "cpu_stack" in row["source"]:
            score += 0.002
        grouped[row["pid"]].append((score, row["source"], row["family"]))
    chosen, chosen_src = [], []
    for pid in pids:
        best = max(grouped[pid], key=lambda t: t[0])
        _, src, fam = best
        chosen.append(fam)
        chosen_src.append(src)
    yt = [truth[p] for p in pids]
    return {
        "n": len(pids),
        "macro_f1": macro(yt, chosen),
        "acc": sum(a == b for a, b in zip(yt, chosen)) / len(yt),
        "chosen_sources": Counter(chosen_src).most_common(20),
    }


def main():
    X, y, val_sources = build_val_dataset()
    print(f"[router] val candidates={len(y)} positive={y.mean():.3f} sources={len(val_sources)}")
    for src in val_sources:
        print(f"[router] train_source={src}", flush=True)
    split_cache = {}
    for split in ["random_full", "drug_cold", "pair_cold"]:
        print(f"[router] loading candidates for {split}", flush=True)
        split_cache[split] = prepare_split(split)
        print(f"[router] {split} pairs={len(split_cache[split][1])}", flush=True)
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "gb": GradientBoostingClassifier(random_state=7, n_estimators=120, max_depth=3, learning_rate=0.05),
    }
    results = {}
    for name, model in models.items():
        model.fit(X, y)
        p = model.predict_proba(X)[:, 1]
        try:
            auc = roc_auc_score(y, p)
        except Exception:
            auc = float("nan")
        results[name] = {"val_auc": auc, "splits": {}}
        print(f"[router] {name} val_auc={auc:.3f}")
        for split in ["random_full", "drug_cold", "pair_cold"]:
            res = eval_router(model, split_cache[split])
            res["split"] = split
            results[name]["splits"][split] = res
            print(f"  {split}: f1={res['macro_f1']:.4f} acc={res['acc']:.4f} n={res['n']}")
    out = OUT / "learned_router_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[router] wrote {out}")

    lines = ["# Learned Student Router Results\n\n"]
    lines.append("Trained candidate-level classifiers on `random_full_val5k`, including locally generated rich CPU variants from rerank candidates, then applied to rich test candidate pools.\n\n")
    lines.append("| Router | Val AUC | random_full | drug_cold | pair_cold | Mean |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for name, r in results.items():
        vals = [r["splits"][s]["macro_f1"] for s in ["random_full", "drug_cold", "pair_cold"]]
        lines.append(f"| {name} | {r['val_auc']:.3f} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {np.mean(vals):.4f} |\n")
    (OUT / "learned_router_results.md").write_text("".join(lines))


if __name__ == "__main__":
    main()
