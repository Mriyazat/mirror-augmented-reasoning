"""Taxonomy Hierarchy Score (THS) — level-aware partial-credit metric.

DDI labels are hierarchical: family → subtype → direction (polarity).
A prediction that gets the family right but the subtype wrong is strictly
better than one that gets the family wrong, and flat macro-F1 hides this.
THS assigns partial credit at each level.

Per-pair score (default weights from the metric specification):
    1.0   full match: family AND subtype AND direction
    0.7   subtype match (family implied correct; direction may be wrong)
    0.3   family-only match (family correct; subtype wrong)
    0.0   family wrong

Macro-THS = simple mean over pairs, stratified by TRUE family so rare
families aren't drowned by AdverseRisk (42.7 % prior).

Direction scoring:
- For bidirectional rows (symmetric templates), direction is trivially
  matched if both predicted+true are bidirectional, else mismatch.
- For directional rows, direction matches iff the (subject, object) pair
  is the same.  Polarity (up / down / risk / risk_down) must also match.

Drop reason always = None.  We assume labels are pre-filtered (no 'Other'
in the labels_hierarchical.parquet).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

# Partial-credit weights . All pairs sum interpretation: mean over
# pairs of the max-level achieved.
W_FAMILY_ONLY = 0.3
W_SUBTYPE = 0.7   # subtype correct (implies family correct)
W_FULL = 1.0      # subtype + direction + polarity all correct


@dataclass(frozen=True)
class Prediction:
    family: str
    subtype: str
    direction: str    # "a_to_b" | "b_to_a" | "bidirectional"
    polarity: str | None  # up / down / risk / risk_down / None for bidirectional AdverseRisk


@dataclass(frozen=True)
class Truth:
    family: str
    subtype: str
    # Direction is implicit in (bidirectional, subject_drugbank_id, object_drugbank_id)
    bidirectional: bool
    subject_drugbank_id: str | None
    object_drugbank_id: str | None
    polarity: str | None
    a_id: str
    b_id: str


def _direction_matches(pred: Prediction, truth: Truth) -> bool:
    """True iff pred direction is consistent with truth direction.

    truth.bidirectional=True  → pred.direction must be "bidirectional".
    truth directional         → pred.direction must resolve to truth's
                                (subject → object) orientation.
    """
    if truth.bidirectional:
        return pred.direction == "bidirectional"
    # Directional case
    if pred.direction == "bidirectional":
        return False
    if pred.direction == "a_to_b":
        return truth.subject_drugbank_id == truth.a_id \
            and truth.object_drugbank_id == truth.b_id
    if pred.direction == "b_to_a":
        return truth.subject_drugbank_id == truth.b_id \
            and truth.object_drugbank_id == truth.a_id
    return False


def _polarity_matches(pred: Prediction, truth: Truth) -> bool:
    return (pred.polarity or None) == (truth.polarity or None)


def score_pair(pred: Prediction, truth: Truth) -> float:
    """Partial-credit THS for a single pair.  Returns 0.0, 0.3, 0.7, or 1.0."""
    if pred.family != truth.family:
        return 0.0
    if pred.subtype != truth.subtype:
        return W_FAMILY_ONLY
    if not _direction_matches(pred, truth) or not _polarity_matches(pred, truth):
        return W_SUBTYPE
    return W_FULL


def score_many(
    preds: Iterable[Prediction],
    truths: Iterable[Truth],
) -> dict:
    """Compute macro-THS + per-family THS + level histogram.

    Returns:
        {
            "macro_ths": float,
            "weighted_ths": float,        # same as micro/simple mean
            "per_family_ths": {fam: float},
            "level_fractions": {
                "level_0_family_wrong": float,
                "level_1_family_only":  float,
                "level_2_subtype":      float,
                "level_3_full":         float,
            },
            "n": int,
        }
    """
    preds = list(preds)
    truths = list(truths)
    assert len(preds) == len(truths), "preds and truths must align"

    scores: list[float] = []
    per_family: dict[str, list[float]] = defaultdict(list)
    level_counts = {
        "level_0_family_wrong": 0,
        "level_1_family_only":  0,
        "level_2_subtype":      0,
        "level_3_full":         0,
    }

    for p, t in zip(preds, truths):
        s = score_pair(p, t)
        scores.append(s)
        per_family[t.family].append(s)
        if s == 0.0:
            level_counts["level_0_family_wrong"] += 1
        elif s == W_FAMILY_ONLY:
            level_counts["level_1_family_only"] += 1
        elif s == W_SUBTYPE:
            level_counts["level_2_subtype"] += 1
        else:
            level_counts["level_3_full"] += 1

    n = len(scores)
    weighted = sum(scores) / n if n else 0.0
    per_family_ths = {f: sum(v) / len(v) for f, v in per_family.items()}
    macro = sum(per_family_ths.values()) / len(per_family_ths) if per_family_ths else 0.0
    level_fractions = {k: v / n for k, v in level_counts.items()} if n else level_counts

    return {
        "macro_ths": macro,
        "weighted_ths": weighted,
        "per_family_ths": per_family_ths,
        "level_fractions": level_fractions,
        "n": n,
    }


# ----------------------------------------------------------------------------
# Convenience: build Truth list from labels_hierarchical.parquet rows.
# ----------------------------------------------------------------------------
def truths_from_rows(rows) -> list[Truth]:
    return [
        Truth(
            family=r["family"],
            subtype=r["subtype"],
            bidirectional=bool(r["bidirectional"]),
            subject_drugbank_id=r["subject_drugbank_id"],
            object_drugbank_id=r["object_drugbank_id"],
            polarity=r["polarity"],
            a_id=r["a_id"],
            b_id=r["b_id"],
        )
        for r in rows
    ]


# ----------------------------------------------------------------------------
# Family-only convenience: XGBoost baseline only predicts FAMILY.  When we
# score a family-only classifier with THS we collapse each pair to the
# family-only level: its THS is W_FAMILY_ONLY if family right, 0 if wrong.
# This lets us compute a THS-ready baseline number today — the eventual
# student (subtype + direction) will be scored with the full pipeline.
# ----------------------------------------------------------------------------
def score_family_only(
    pred_families,   # list[str]
    truth_families,  # list[str]
) -> dict:
    assert len(pred_families) == len(truth_families)
    per_family: dict[str, list[float]] = defaultdict(list)
    level_counts = {
        "level_0_family_wrong": 0,
        "level_1_family_only":  0,
        "level_2_subtype":      0,
        "level_3_full":         0,
    }
    for p, t in zip(pred_families, truth_families):
        s = W_FAMILY_ONLY if p == t else 0.0
        per_family[t].append(s)
        if s == 0.0:
            level_counts["level_0_family_wrong"] += 1
        else:
            level_counts["level_1_family_only"] += 1

    n = len(pred_families)
    weighted = sum(sum(v) for v in per_family.values()) / n if n else 0.0
    per_family_ths = {f: sum(v) / len(v) for f, v in per_family.items()}
    macro = sum(per_family_ths.values()) / len(per_family_ths) if per_family_ths else 0.0
    return {
        "macro_ths": macro,
        "weighted_ths": weighted,
        "per_family_ths": per_family_ths,
        "level_fractions": {k: v / n for k, v in level_counts.items()} if n else level_counts,
        "n": n,
        "note": "family-only scorer (max possible per pair = 0.3 since subtype/direction not predicted)",
    }
