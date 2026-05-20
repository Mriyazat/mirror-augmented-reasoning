"""CSA -- Cross-Symmetry Agreement (, complements MPS).

Definition
----------
    CSA = P(correct | order (A, B))  AND  correct | order (B, A)

    i.e. the joint probability that the model predicts the gold family
    correctly under BOTH input orderings of a mirror pair.

Where CSA differs from MPS
--------------------------
MPS asks: "Given family is correct in both orderings, does the direction
tag FLIP correctly?"  It measures direction symmetry.

CSA asks: "Does family correctness SURVIVE the order swap?"  It
measures family-prediction stability.  A model can have high MPS (good
direction handling) but low CSA (it gets family right in one ordering
and wrong in the other).  The paper table reports both to separate
these failure modes.

Ablation interpretation
-----------------------
With independent errors:      CSA = acc_ab * acc_ba    (product baseline).
With perfect symmetry:        CSA = min(acc_ab, acc_ba).
Observed CSA vs the product baseline tells us whether errors are
ordering-correlated:
    CSA > product  -> model is symmetric; getting ab right helps ba.
    CSA = product  -> orderings are independent.
    CSA < product  -> order-dependent failures (a known failure pattern -- hidden
                       in flat accuracy tables until CSA exposes them).

Design
------
Input records carry both predictions plus the gold family.  We report:
    csa               = P(correct in both)
    product_baseline  = P(correct | ab) * P(correct | ba)
    csa_lift          = csa - product_baseline
    consistency       = P(pred_ab.family == pred_ba.family)
                         regardless of gold  (pure symmetry signal)

Usage
-----
    from src.metrics.csa import CsaRecord, csa_corpus
    rep = csa_corpus(records)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CsaRecord:
    pair_id: str
    gold_family: str
    pred_family_ab: str | None
    pred_family_ba: str | None


def csa_corpus(records: Iterable[CsaRecord]) -> dict:
    """Compute CSA + diagnostic companions."""
    n_total = 0
    n_ab_ok = 0
    n_ba_ok = 0
    n_both_ok = 0
    n_consistent = 0       # pred_ab.family == pred_ba.family (regardless of correctness)
    by_fam_total: dict[str, int] = defaultdict(int)
    by_fam_both: dict[str, int] = defaultdict(int)

    for r in records:
        n_total += 1
        by_fam_total[r.gold_family] += 1
        ab_ok = r.pred_family_ab == r.gold_family
        ba_ok = r.pred_family_ba == r.gold_family
        if ab_ok:
            n_ab_ok += 1
        if ba_ok:
            n_ba_ok += 1
        if ab_ok and ba_ok:
            n_both_ok += 1
            by_fam_both[r.gold_family] += 1
        if r.pred_family_ab is not None and r.pred_family_ab == r.pred_family_ba:
            n_consistent += 1

    if n_total == 0:
        return {
            "csa":               0.0,
            "acc_ab":            0.0,
            "acc_ba":            0.0,
            "product_baseline":  0.0,
            "csa_lift":          0.0,
            "consistency":       0.0,
            "n_total":           0,
            "per_family_csa":    {},
            "per_family_n":      {},
        }

    acc_ab = n_ab_ok / n_total
    acc_ba = n_ba_ok / n_total
    csa = n_both_ok / n_total
    product = acc_ab * acc_ba
    consistency = n_consistent / n_total

    per_family = {
        f: (by_fam_both[f] / by_fam_total[f]) for f in by_fam_total
    }

    return {
        "csa":               csa,
        "acc_ab":            acc_ab,
        "acc_ba":            acc_ba,
        "product_baseline":  product,
        "csa_lift":          csa - product,
        "consistency":       consistency,
        "n_total":           n_total,
        "n_both_ok":         n_both_ok,
        "n_ab_ok":           n_ab_ok,
        "n_ba_ok":           n_ba_ok,
        "n_consistent":      n_consistent,
        "per_family_csa":    per_family,
        "per_family_n":      dict(by_fam_total),
    }
