"""AU -- Abstention Utility.
Definition
----------
    AU = (N_correct_kept - alpha * N_incorrect_kept) / N_total

where
    N_correct_kept    = # predictions that were KEPT (not abstained)
                        and were CORRECT,
    N_incorrect_kept  = # predictions that were KEPT and were WRONG,
    N_total           = # predictions total (including abstentions),
    alpha             = cost ratio: how bad is one wrong answer versus
                        how good is one right answer.

Choice of alpha
---------------
For DDI the clinical cost of a false mechanism prediction is
significantly higher than the marginal loss of an abstention.  We
default alpha = 1.0 (the breakeven Chow cost) for the main paper table,
and additionally report alpha in {2.0, 5.0} for severity-stratified
robustness.  alpha > 1 means "a wrong prediction is worth alpha right
predictions worth of disutility".

Success criteria
----------------
    - AU > 0 at 90 % coverage (i.e. abstain on at most 10 % of pairs).

Related metrics we also expose
------------------------------
- **Selective accuracy** = N_correct_kept / (N_kept) -- the usual
  conditional accuracy.
- **Coverage**           = N_kept / N_total.
- **Risk-coverage curve** = (coverage, error_rate) pairs as the
  abstention threshold sweeps from 0 to 1.  Area under the risk-coverage
  curve (AURC) is a single-number summary; lower is better.

Design choices
--------------
1.  Two usage modes:
    (a) **Single-threshold AU** -- caller has already decided which pairs
        to abstain on, passes booleans.
    (b) **Score-threshold AU**  -- caller passes a confidence score per
        pair; we sweep thresholds and compute the whole curve.

2.  **Abstention = honest-answer or forced-abstention flag.**  Either an
    explicit `{"abstain": True}` final_answer or a confidence below the
    threshold counts as abstention.  Errors on abstained pairs are zero
    by definition.

3.  **Gold comparison**: the caller supplies `pred_correct: bool` per
    pair.  This decouples AU from the taxonomy-level metric used for
    correctness (family-only, subtype, full hierarchy).  For specification
    evaluation we compute AU using family-correct-ness.

Input data model (single-threshold)
-----------------------------------
    AbstentionRecord(
        pair_id:              str,
        pred_correct:         bool,          # True iff the (kept) pred is right
        abstained:            bool,
        confidence:           float | None,  # optional (for score mode)
        gold_family:          str | None,    # optional (for stratified report)
        severity:             str | None,    # optional (for DDInter stratification)
    )

Typical usage
-------------
    from src.metrics.au import AbstentionRecord, au_single, au_curve
    rep = au_single(records, alpha=1.0)
    curve = au_curve(records)     # requires confidence scores
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AbstentionRecord:
    pair_id: str
    pred_correct: bool
    abstained: bool
    confidence: float | None = None
    gold_family: str | None = None
    severity: str | None = None


def au_single(records: Iterable[AbstentionRecord],
              alpha: float = 1.0) -> dict:
    """Single-threshold AU: abstentions are already decided.

    Returns a report dict with stratifications by family and severity
    (when present in the records).
    """
    n_total = n_abstained = n_correct = n_incorrect = 0
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    # per-family: [n_total, n_abstained, n_correct, n_incorrect]
    by_sev: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])

    for r in records:
        n_total += 1
        f = r.gold_family or "UNKNOWN"
        s = r.severity or "UNKNOWN"
        by_family[f][0] += 1
        by_sev[s][0] += 1
        if r.abstained:
            n_abstained += 1
            by_family[f][1] += 1
            by_sev[s][1] += 1
            continue
        if r.pred_correct:
            n_correct += 1
            by_family[f][2] += 1
            by_sev[s][2] += 1
        else:
            n_incorrect += 1
            by_family[f][3] += 1
            by_sev[s][3] += 1

    n_kept = n_correct + n_incorrect
    coverage = (n_kept / n_total) if n_total else 0.0
    selective_acc = (n_correct / n_kept) if n_kept else 0.0
    au = ((n_correct - alpha * n_incorrect) / n_total) if n_total else 0.0

    def _stratum_au(counts: list[int]) -> dict:
        t, a, c, i = counts
        k = c + i
        return {
            "n_total":        t,
            "n_abstained":    a,
            "n_correct":      c,
            "n_incorrect":    i,
            "coverage":       (k / t) if t else 0.0,
            "selective_acc":  (c / k) if k else 0.0,
            "au":             ((c - alpha * i) / t) if t else 0.0,
        }

    return {
        "au":               au,
        "alpha":            alpha,
        "coverage":         coverage,
        "selective_acc":    selective_acc,
        "n_total":          n_total,
        "n_abstained":      n_abstained,
        "n_correct":        n_correct,
        "n_incorrect":      n_incorrect,
        "per_family":       {f: _stratum_au(c) for f, c in by_family.items()},
        "per_severity":     {s: _stratum_au(c) for s, c in by_sev.items()},
    }


def au_curve(records: Sequence[AbstentionRecord],
             alpha: float = 1.0) -> dict:
    """Score-threshold AU: sweep confidence threshold and trace the
    risk-coverage curve plus AU at each coverage level.

    Every record must have a non-None `confidence`.  Records whose
    `abstained=True` at confidence=None are treated as abstaining at every
    threshold (always abstain).

    Returns:
        {
            "curve": [
                { "threshold": float, "coverage": float,
                  "selective_acc": float, "au": float,
                  "n_kept": int, "n_correct": int, "n_incorrect": int },
                ...
            ],
            "au_at_90_coverage":        float,   # main specification number
            "au_at_coverage": { 0.5: ..., 0.8: ..., 0.9: ..., 0.95: ... },
            "aurc":                     float,   # area under risk-coverage curve
            "best_au":                  float,
            "best_threshold":           float,
        }

    `au_at_90_coverage` = AU at the threshold that gives the highest
    coverage <= 0.90.  If coverage never drops below 0.90 (all
    predictions kept) this equals AU at coverage=1.
    """
    recs = [r for r in records if r.confidence is not None or r.abstained]
    n_total = len(recs)
    if n_total == 0:
        return {
            "curve": [],
            "au_at_90_coverage": 0.0,
            "au_at_coverage": {},
            "aurc": 0.0,
            "best_au": 0.0,
            "best_threshold": 0.0,
        }

    # Sort records by confidence descending.  Records that unconditionally
    # abstain are placed at the bottom (lowest confidence).
    def _conf(r: AbstentionRecord) -> float:
        if r.abstained and r.confidence is None:
            return float("-inf")
        return r.confidence if r.confidence is not None else float("-inf")

    sorted_recs = sorted(recs, key=_conf, reverse=True)

    # Sweep: keep top-k predictions (by confidence), abstain the rest.
    curve: list[dict] = []
    cum_c = cum_i = 0
    thresholds_at: dict[float, float] = {}
    target_coverages = [0.50, 0.80, 0.90, 0.95]

    aurc_accum = 0.0
    prev_cov = 0.0
    prev_err = 0.0

    for k in range(1, n_total + 1):
        r = sorted_recs[k - 1]
        if r.abstained and r.confidence is None:
            # Unconditional abstention -- don't include it in "kept"
            # Only can happen at the tail; handle via subsequent points.
            continue
        if r.pred_correct:
            cum_c += 1
        else:
            cum_i += 1
        cov = k / n_total
        sel_acc = (cum_c / (cum_c + cum_i)) if (cum_c + cum_i) else 0.0
        au = (cum_c - alpha * cum_i) / n_total
        err = 1.0 - sel_acc
        # Trapezoidal AURC
        aurc_accum += 0.5 * (cov - prev_cov) * (err + prev_err)
        prev_cov, prev_err = cov, err

        point = {
            "threshold":     _conf(r),
            "coverage":      cov,
            "selective_acc": sel_acc,
            "au":            au,
            "n_kept":        cum_c + cum_i,
            "n_correct":     cum_c,
            "n_incorrect":   cum_i,
        }
        curve.append(point)

    # AU at specific coverage targets
    for target in target_coverages:
        best_match = None
        for pt in curve:
            if pt["coverage"] <= target + 1e-9:
                if best_match is None or pt["coverage"] > best_match["coverage"]:
                    best_match = pt
        thresholds_at[target] = best_match["au"] if best_match else 0.0

    best_pt = max(curve, key=lambda p: p["au"]) if curve else None

    return {
        "curve":                curve,
        "au_at_90_coverage":    thresholds_at.get(0.90, 0.0),
        "au_at_coverage":       thresholds_at,
        "aurc":                 aurc_accum,
        "best_au":              best_pt["au"] if best_pt else 0.0,
        "best_threshold":       best_pt["threshold"] if best_pt else 0.0,
        "alpha":                alpha,
    }
