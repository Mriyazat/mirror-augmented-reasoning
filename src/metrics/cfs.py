"""CfS -- Counterfactual Sensitivity .

Definition
----------
    CfS = KL( P(y | original PK) || P(y | perturbed PK) )
          averaged over PK perturbations that SHOULD change the mechanism.

    Sibling number, reported alongside:
    CfS_null  = same KL averaged over perturbations that should NOT change
                 the mechanism (nuisance floor).

Motivation
----------
earlier baselines got away with label-co-occurrence shortcuts: the model
could predict CYP3A4 metabolism interactions from drug names without
ever relying on the PK flags.  Our claim is that the model is mechanism-aware,
not surface-pattern-aware.  A mechanism-aware model MUST change its
prediction when you flip the PK flag that drives the mechanism, and it
MUST NOT change when you flip an unrelated flag.  CfS measures both.

If CfS is LOW on "relevant" perturbations, the model is ignoring PK.
If CfS is HIGH on "irrelevant" perturbations, the model is chasing
nuisance features.  The informative number is the GAP
    CfS_relevant - CfS_null.

Design choices
--------------
1.  **Symmetric KL optional.**  By default we report one-sided KL(P_orig
    || P_perturbed) because that is what the plan says.  A `symmetric=True`
    flag gives the Jensen-Shannon-divergence variant, which is what recent
    literature (Kadavath et al. 2022) prefers for probe interpretations.

2.  **Epsilon smoothing.**  Both distributions are additively smoothed by
    a tiny eps (default 1e-9) to keep KL finite when either assigns 0 to
    a class.  Eps is renormalized after smoothing so distributions still
    sum to 1.

3.  **Domain agnostic.**  The metric consumes ANY pair of label
    distributions.  For family-only predictions the label space is the
    8 families; for fine-grained it could be the 100-subtype space.
    The caller controls granularity.

4.  **Per-perturbation stratification.**  Different PK flags have
    different expected effect sizes (flipping CYP3A4 flag matters more
    than flipping OCT1).  We keep per-flag rates in the report so the
    paper table can highlight which perturbations the model is sensitive
    to.

Input data model
----------------
    CounterfactualRecord(
        pair_id:               str,
        perturbation:          str,           # name of the flag flipped
        relevant:              bool,          # True if this flip SHOULD change
                                              # the mechanism (per gold)
        p_original:            dict[str, float],  # P over labels, orig context
        p_perturbed:           dict[str, float],  # P over labels, perturbed ctx
    )

Usage
-----
    from src.metrics.cfs import CounterfactualRecord, cfs_corpus
    recs = [...]
    report = cfs_corpus(recs)
    print(report["cfs_relevant"], report["cfs_null"], report["cfs_gap"])
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

_EPS_DEFAULT = 1e-9


@dataclass(frozen=True)
class CounterfactualRecord:
    pair_id: str
    perturbation: str
    relevant: bool
    p_original: dict[str, float]
    p_perturbed: dict[str, float]


def _align_and_smooth(p: dict[str, float], q: dict[str, float],
                      eps: float) -> tuple[list[float], list[float]]:
    """Align two dicts to a common label set, add eps, renormalize."""
    labels = set(p) | set(q)
    pv = [max(p.get(l, 0.0), 0.0) + eps for l in labels]
    qv = [max(q.get(l, 0.0), 0.0) + eps for l in labels]
    ps = sum(pv); qs = sum(qv)
    pv = [x / ps for x in pv]
    qv = [x / qs for x in qv]
    return pv, qv


def _kl(pv: list[float], qv: list[float]) -> float:
    """KL(P || Q) in nats.  Both distributions must be smoothed & normalized."""
    return sum(p * math.log(p / q) for p, q in zip(pv, qv) if p > 0)


def _jsd(pv: list[float], qv: list[float]) -> float:
    """Jensen-Shannon divergence in nats, bounded [0, ln 2]."""
    mv = [0.5 * (p + q) for p, q in zip(pv, qv)]
    return 0.5 * _kl(pv, mv) + 0.5 * _kl(qv, mv)


def cfs_pair(p_orig: dict[str, float], p_pert: dict[str, float],
             symmetric: bool = False, eps: float = _EPS_DEFAULT) -> float:
    """Per-counterfactual divergence between two label distributions."""
    pv, qv = _align_and_smooth(p_orig, p_pert, eps)
    return _jsd(pv, qv) if symmetric else _kl(pv, qv)


def cfs_corpus(records: Iterable[CounterfactualRecord],
               symmetric: bool = False,
               eps: float = _EPS_DEFAULT) -> dict:
    """Aggregate CfS over a corpus of counterfactual records.

    Returns:
        {
            "cfs_relevant":           float,   # avg KL on 'relevant' flips
            "cfs_null":               float,   # avg KL on 'irrelevant' flips
            "cfs_gap":                float,   # relevant - null  (main headline)
            "n_relevant":             int,
            "n_null":                 int,
            "per_perturbation": {
                flag: {"n_rel": int, "cfs_rel": float,
                       "n_null": int, "cfs_null": float},
            },
            "symmetric":              bool,
            "eps":                    float,
        }

    A positive `cfs_gap` >= the metric specification target (0.20) is the success
    criterion.  A gap near zero means the model ignores PK flags.  A
    negative gap (rare) means the model is confused by irrelevant
    flips more than it is sensitive to relevant ones -- a red flag.
    """
    rel_scores: list[float] = []
    null_scores: list[float] = []
    by_flag_rel: dict[str, list[float]] = defaultdict(list)
    by_flag_null: dict[str, list[float]] = defaultdict(list)

    for r in records:
        d = cfs_pair(r.p_original, r.p_perturbed, symmetric=symmetric, eps=eps)
        if r.relevant:
            rel_scores.append(d)
            by_flag_rel[r.perturbation].append(d)
        else:
            null_scores.append(d)
            by_flag_null[r.perturbation].append(d)

    cfs_rel = (sum(rel_scores) / len(rel_scores)) if rel_scores else 0.0
    cfs_null = (sum(null_scores) / len(null_scores)) if null_scores else 0.0

    per_flag: dict[str, dict] = {}
    for flag in set(by_flag_rel) | set(by_flag_null):
        rs = by_flag_rel.get(flag, [])
        ns = by_flag_null.get(flag, [])
        per_flag[flag] = {
            "n_rel":    len(rs),
            "cfs_rel":  (sum(rs) / len(rs)) if rs else 0.0,
            "n_null":   len(ns),
            "cfs_null": (sum(ns) / len(ns)) if ns else 0.0,
        }

    return {
        "cfs_relevant":    cfs_rel,
        "cfs_null":        cfs_null,
        "cfs_gap":         cfs_rel - cfs_null,
        "n_relevant":      len(rel_scores),
        "n_null":          len(null_scores),
        "per_perturbation": per_flag,
        "symmetric":       symmetric,
        "eps":             eps,
    }
