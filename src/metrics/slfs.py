"""SLFS -- Step-Level Faithfulness Score.
Definition
----------
    SLFS = mean DDI-PRM step score on the model's reasoning steps.

Scope
-----
SLFS is the headline metric for DDI-PRM (our domain-specialized Process
Reward Model, novelty P3+P6).  It summarizes, per trace, how faithful
each step of the rationale is to the rubric dimensions:
    (relevance, factuality, direction-preservation, PK-consistency,
     conclusion-match).

The DDI-PRM outputs a per-step scalar in [0, 1] for each rubric
dimension; SLFS aggregates these into a trace-level and then corpus-
level score.  At trace level we default to MEAN over steps; at corpus
level we default to MACRO-over-families (matching our other metrics).

Design choices
--------------
1.  **PRM-agnostic.**  This file does NOT depend on the PRM model or
    weights.  It consumes traces where each step already has a
    `prm_score` field (scalar or dict of dimension -> scalar).  The
    actual PRM inference lives in `src/teacher/prm_train.py` + the
    forthcoming `src/teacher/prm_score.py` runner.

2.  **Dimension reduction.**  If step has a dict of dimension scores,
    SLFS defaults to the unweighted MEAN of the rubric dimensions.
    Callers can pass `dim_weights={dim: w}` to weight specific rubric
    aspects (e.g. 2x direction-preservation for ablations targeting
    the mirror-DPO pillar).

3.  **Meta step handling.**  Terminal/meta steps (`conclusion`,
    `abstention`, `evidence_gap`) are by default INCLUDED with whatever
    score the PRM gave them -- the PRM is trained to evaluate these too
    (e.g. an `abstention` step should score high when evidence is
    genuinely missing).  Pass `exclude_meta=True` to restrict SLFS to
    rationale-only steps if you want the PK/mechanism signal alone.

4.  **Min-plus aggregation (optional).**  The PRM-guided critic uses "min-plus"
    aggregation (min of dimension scores plus a small reward for the
    second-worst, used to penalize the one-bad-step failure mode).
    We expose this via `aggregator="min_plus"`.  Default is "mean".

Usage
-----
    from src.metrics.slfs import slfs_trace, slfs_corpus

    score = slfs_trace(trace_with_prm_scores)
    report = slfs_corpus(records)   # records have 'trace' + optional 'family'
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence


_META_ROLES = {"conclusion", "abstention", "evidence_gap"}


def _extract_step_score(step: dict,
                        dim_weights: dict[str, float] | None) -> float | None:
    """Pull the PRM score off a step dict.  Handles both shapes:
        step["prm_score"] = 0.82
        step["prm_score"] = {"relevance": 0.9, "factuality": 0.8, ...}
    """
    val = step.get("prm_score")
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        if not val:
            return None
        if dim_weights:
            total_w = 0.0
            acc = 0.0
            for dim, w in dim_weights.items():
                if dim in val and isinstance(val[dim], (int, float)):
                    acc += w * float(val[dim])
                    total_w += w
            if total_w == 0.0:
                return None
            return acc / total_w
        # unweighted mean over present numeric dims
        nums = [float(v) for v in val.values() if isinstance(v, (int, float))]
        if not nums:
            return None
        return sum(nums) / len(nums)
    return None


def slfs_trace(trace: dict,
               aggregator: str = "mean",
               exclude_meta: bool = False,
               dim_weights: dict[str, float] | None = None) -> float:
    """Trace-level SLFS.

    aggregator:
        "mean"      -- arithmetic mean over steps (default)
        "min_plus"  -- min + 0.1 * second_min  (penalize single-bad-step)
        "min"       -- worst step score (strictest)

    exclude_meta:
        when True, drop conclusion/abstention/evidence_gap steps before
        aggregating.
    """
    steps = (trace.get("steps") or []) if isinstance(trace, dict) else []
    scores: list[float] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        if exclude_meta and s.get("role") in _META_ROLES:
            continue
        v = _extract_step_score(s, dim_weights)
        if v is not None:
            scores.append(v)
    if not scores:
        return 0.0
    if aggregator == "mean":
        return sum(scores) / len(scores)
    if aggregator == "min":
        return min(scores)
    if aggregator == "min_plus":
        s_sorted = sorted(scores)
        m1 = s_sorted[0]
        m2 = s_sorted[1] if len(s_sorted) > 1 else m1
        return m1 + 0.1 * m2
    raise ValueError(f"slfs_trace: unknown aggregator {aggregator!r}")


def slfs_corpus(records: Sequence[dict],
                aggregator: str = "mean",
                exclude_meta: bool = False,
                dim_weights: dict[str, float] | None = None) -> dict:
    """Aggregate SLFS over a corpus.

    Each record:
        { "trace": <dict>, "family": <str>?, "trace_id": <str>? }
    """
    rows: list[tuple[str, float]] = []
    for rec in records:
        tr = rec.get("trace") if isinstance(rec, dict) else None
        if not isinstance(tr, dict):
            continue
        fam = rec.get("family") or "UNKNOWN"
        s = slfs_trace(tr, aggregator=aggregator, exclude_meta=exclude_meta,
                       dim_weights=dim_weights)
        rows.append((fam, s))

    n = len(rows)
    weighted = (sum(s for _, s in rows) / n) if n else 0.0

    by_fam: dict[str, list[float]] = defaultdict(list)
    for fam, s in rows:
        by_fam[fam].append(s)
    per_family = {f: (sum(v) / len(v) if v else 0.0) for f, v in by_fam.items()}
    macro = (sum(per_family.values()) / len(per_family)) if per_family else 0.0

    return {
        "slfs":            macro,         # primary reported number (macro)
        "weighted_slfs":   weighted,
        "per_family_slfs": per_family,
        "per_family_n":    {f: len(v) for f, v in by_fam.items()},
        "n":               n,
        "aggregator":      aggregator,
        "exclude_meta":    exclude_meta,
        "dim_weights":     dim_weights,
    }
