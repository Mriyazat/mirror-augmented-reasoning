"""Per-family conformal abstention calibration for the student.

Reads a *probe* predictions file (predictions on a held-out probe set with
ground-truth labels available), computes a per-family confidence threshold
$\tau_f$ such that on the probe set the family-conditional accuracy of
non-abstained predictions is at least the user-specified target (default
0.85), and writes the threshold table to JSON for inference-time use.

Why per-family
--------------
The headline failure on the held-out test is one rare class
(PK_Distribution: sel-acc 0.262) dragging macro-F1 down.  A single global
threshold is too blunt: it either abstains too much on the easy classes
(AdverseRisk, sel-acc 0.77) or too little on the hard ones.  A family-
specific threshold lets us refuse-rather-than-guess on PK_Distribution
while keeping coverage on AdverseRisk near 1.

Output schema (`thresholds.json`)
---------------------------------
    {
      "target_family_accuracy": 0.85,
      "metric":     "geomean_plus",   # which PRM aggregation we threshold
      "global_tau": 0.62,
      "per_family": {
          "AdverseRisk":     0.31,
          "Efficacy":        0.71,
          "PD_Activity":     0.66,
          "PK_Excretion":    0.62,
          "PK_Metabolism":   0.49,
          "PK_Distribution": 0.78,
          "PK_Absorption":   0.55
      },
      "family_audit": {
          "AdverseRisk": {"n":1430, "sel_acc_at_tau":0.85, "coverage":0.91},
          ...
      }
    }

The applier (`apply_conformal_thresholds`) takes a predictions JSONL with
per-record PRM scores (e.g. the rerank output) and rewrites
`final_prediction.abstain` according to the table.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_jsonl(path: str | Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_truth(labels_parquet: str | Path) -> dict[str, dict]:
    """pair_id -> {family, subtype, direction_tag(if available)}."""
    import pyarrow.parquet as pq
    rows = pq.read_table(
        labels_parquet,
        columns=["pair_id", "family", "subtype", "bidirectional"],
    ).to_pylist()
    out = {}
    for r in rows:
        pid = r["pair_id"]
        out[pid] = {
            "family":  r.get("family"),
            "subtype": r.get("subtype"),
            "direction_tag": "bidirectional"
                if r.get("bidirectional") else None,
        }
    return out


_METRIC_TO_CAND_FIELD = {
    "geomean_plus": "prm_geomean",
    "mean_plus":    "prm_mean",
    "min_plus":     "prm_min",
    "final_plus":   "prm_final",
}


def _record_score(rec: dict, metric: str) -> float | None:
    """Pull a PRM score out of a prediction record.  Looks in three places
    in order of preference:
      1. rec['rerank']['chosen_prm']  (from predict_with_rerank when its
         own --rerank_metric matches `metric`)
      2. rec['rerank']['candidates'][chosen_idx][prm_<suffix>]
      3. rec['final_prediction']['confidence']  (LLM self-reported)
    Returns None if no score is available.
    """
    rer = rec.get("rerank") or {}
    # If the rerank pass used the same metric we're calibrating on, we can
    # trust chosen_prm directly.  Otherwise fall through to the per-candidate
    # field so we don't accidentally calibrate on a metric the rerank
    # didn't optimise.
    if rer.get("metric") == metric and "chosen_prm" in rer:
        return float(rer["chosen_prm"])
    cands = rer.get("candidates") or []
    idx = rer.get("chosen_idx", -1)
    if 0 <= idx < len(cands):
        field = _METRIC_TO_CAND_FIELD.get(metric, "prm_geomean")
        v = cands[idx].get(field)
        if v is not None:
            return float(v)
    fp = rec.get("final_prediction") or {}
    c = fp.get("confidence")
    if c is not None:
        try:
            return float(c)
        except Exception:
            return None
    return None


def _calibrate(probe_records: list[dict],
               truth: dict[str, dict],
               metric: str,
               target_acc: float,
               min_coverage: float = 0.5
               ) -> dict:
    """For each family, find the smallest threshold tau such that the
    accuracy among predictions with PRM score >= tau is >= target_acc,
    subject to coverage >= min_coverage.  If no tau achieves the target
    while meeting min_coverage, fall back to the median PRM score for that
    family (i.e. abstain on the bottom 50%)."""
    by_fam: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    n_used = 0
    n_skip = 0
    for rec in probe_records:
        pid = rec.get("pair_id")
        if pid not in truth:
            n_skip += 1
            continue
        fp = rec.get("final_prediction") or {}
        pred_fam = fp.get("family")
        if pred_fam is None:
            continue
        score = _record_score(rec, metric)
        if score is None:
            continue
        gold_fam = truth[pid]["family"]
        correct = (pred_fam == gold_fam)
        by_fam[pred_fam].append((score, correct))
        n_used += 1

    per_family: dict[str, float] = {}
    audit: dict[str, dict] = {}

    for fam, items in by_fam.items():
        if not items:
            continue
        # Sweep candidate thresholds = unique scores in the family
        scores_sorted = sorted({s for s, _ in items}, reverse=True)
        best_tau = None
        n_total = len(items)
        # Pass 1: smallest tau with target_acc AND coverage >= min_coverage
        for tau in scores_sorted:
            kept = [c for s, c in items if s >= tau]
            if len(kept) < min_coverage * n_total:
                continue
            acc = sum(kept) / len(kept) if kept else 0.0
            if acc >= target_acc:
                best_tau = tau  # keep walking to find a lower one (more coverage)
            elif best_tau is not None:
                break
        fallback = "none"
        if best_tau is None:
            # Pass 2: relax min_coverage; pick the smallest tau (most coverage)
            # that still meets target_acc.  Requires at least 10 records to
            # avoid 1-of-1 estimates.
            for tau in scores_sorted:
                kept = [c for s, c in items if s >= tau]
                if len(kept) < 10:
                    continue
                acc = sum(kept) / len(kept) if kept else 0.0
                if acc >= target_acc:
                    best_tau = tau
            if best_tau is not None:
                fallback = "relaxed_coverage"
        if best_tau is None:
            # Pass 3: target unreachable for this family.  Use median (the
            # original safety net) and flag it so the user sees this in audit.
            sorted_scores = sorted([s for s, _ in items])
            best_tau = sorted_scores[len(sorted_scores) // 2]
            fallback = "median_unreachable"

        kept = [c for s, c in items if s >= best_tau]
        per_family[fam] = float(best_tau)
        audit[fam] = {
            "n":              int(n_total),
            "n_kept":         int(len(kept)),
            "coverage":       float(len(kept) / n_total) if n_total else 0.0,
            "sel_acc_at_tau": float(sum(kept) / len(kept)) if kept else 0.0,
            "fallback":       fallback,
        }

    # Global tau over all records
    all_items = [(s, c) for items in by_fam.values() for s, c in items]
    if all_items:
        all_scores = sorted({s for s, _ in all_items}, reverse=True)
        global_tau = None
        for tau in all_scores:
            kept = [c for s, c in all_items if s >= tau]
            if len(kept) < min_coverage * len(all_items):
                continue
            acc = sum(kept) / len(kept)
            if acc >= target_acc:
                global_tau = tau
        if global_tau is None:
            sorted_scores = sorted([s for s, _ in all_items])
            global_tau = sorted_scores[len(sorted_scores) // 2]
    else:
        global_tau = 0.5

    return {
        "metric":                  metric,
        "target_family_accuracy":  float(target_acc),
        "min_coverage":            float(min_coverage),
        "global_tau":              float(global_tau),
        "per_family":              per_family,
        "family_audit":            audit,
        "n_probe_records":         int(n_used),
        "n_probe_skipped":         int(n_skip),
    }


def apply_conformal_thresholds(
    in_path: str | Path,
    out_path: str | Path,
    thresholds_path: str | Path,
    use_global_for_unknown: bool = True,
):
    """Re-write a predictions JSONL by setting `final_prediction.abstain`
    to True whenever the per-family PRM score < tau_f.  Leaves all other
    fields alone."""
    with open(thresholds_path) as f:
        cal = json.load(f)
    metric = cal["metric"]
    per_fam = cal["per_family"]
    global_tau = cal["global_tau"]

    n_in = n_abstained = n_kept = 0
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(in_path) as fin, out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_in += 1
            fp = rec.setdefault("final_prediction", {})
            pred_fam = fp.get("family")
            score = _record_score(rec, metric)
            tau = per_fam.get(pred_fam,
                              global_tau if use_global_for_unknown else 0.0)
            already_abstaining = bool(fp.get("abstain", False))
            new_abstain = already_abstaining or (
                score is None or score < tau
            )
            fp["abstain"] = bool(new_abstain)
            fp["conformal_tau"] = float(tau)
            fp["conformal_score"] = (None if score is None else float(score))
            if new_abstain:
                n_abstained += 1
            else:
                n_kept += 1
            fout.write(json.dumps(rec) + "\n")
    print(f"[conformal] read {n_in:,}; "
          f"abstained {n_abstained:,} ({100*n_abstained/max(n_in,1):.1f}%); "
          f"kept {n_kept:,}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    cal = sub.add_parser("calibrate", help="Build the threshold table.")
    cal.add_argument("--probe_predictions", required=True,
                     help="Predictions JSONL with per-record PRM scores "
                          "(e.g. predict_with_rerank output on a held-out "
                          "probe set).")
    cal.add_argument("--labels", required=True,
                     help="data_processed/labels_hierarchical.parquet")
    cal.add_argument("--metric", default="geomean_plus",
                     choices=["geomean_plus", "min_plus", "final_plus", "mean_plus"])
    cal.add_argument("--target_accuracy", type=float, default=0.85)
    cal.add_argument("--min_coverage", type=float, default=0.5)
    cal.add_argument("--output", required=True,
                     help="Output thresholds.json")

    app = sub.add_parser("apply", help="Apply a threshold table to a "
                                       "predictions file.")
    app.add_argument("--input", required=True)
    app.add_argument("--output", required=True)
    app.add_argument("--thresholds", required=True)
    app.add_argument("--no_global_fallback", action="store_true",
                     help="If a predicted family is not in the calibration "
                          "table, do NOT use the global tau as a fallback.")

    args = ap.parse_args()

    if args.cmd == "calibrate":
        probe_records = list(_load_jsonl(args.probe_predictions))
        truth = _load_truth(args.labels)
        cal_result = _calibrate(
            probe_records, truth, args.metric,
            args.target_accuracy, args.min_coverage,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cal_result, indent=2) + "\n")
        print(f"[conformal] calibrated on {cal_result['n_probe_records']} "
              f"records.")
        print(f"[conformal] global_tau = {cal_result['global_tau']:.3f}")
        for fam, tau in sorted(cal_result["per_family"].items(),
                               key=lambda kv: kv[0]):
            au = cal_result["family_audit"][fam]
            print(f"  {fam:20s} tau={tau:.3f}  "
                  f"cov={au['coverage']:.2f}  "
                  f"sel_acc={au['sel_acc_at_tau']:.3f}  "
                  f"(n={au['n']})")
        print(f"[conformal] wrote {out}")
    elif args.cmd == "apply":
        apply_conformal_thresholds(
            args.input, args.output, args.thresholds,
            use_global_for_unknown=not args.no_global_fallback,
        )


if __name__ == "__main__":
    main()
