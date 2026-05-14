"""MFS -- Mechanism-Faithfulness Score (novelty pillar P5).

Definition (plan §8)
--------------------
    MFS = fraction of rationale claims that cite at least one evidence ID
          which resolves to the retrieved pathway / PK / proteomics context.

Intuition: a trace whose steps cite real evidence from the retrieval bundle
is *mechanism-grounded*; a trace that invents identifiers or relies on
model-internal priors is *not*.  MFS is the continuous cousin of the QC
G2 gate: G2 is a pass/fail per trace, MFS is a score in [0, 1].

Scope
-----
MFS is scored on *any* DDI reasoning trace -- teacher candidates, student
predictions, or baseline outputs -- so long as the trace conforms to the
rubric schema in `src.teacher.schema` (steps + final_answer).

Design choices
--------------
1.  **Step-level granularity.**  Each step contributes one vote to the
    fraction.  Steps with non-empty `evidence_ids` AND at least one
    resolvable id score 1.  Steps with empty `evidence_ids` or only
    invented ids score 0.

2.  **Meta roles excluded.**  Steps whose role is `conclusion`,
    `abstention`, or `evidence_gap` do not assert a mechanistic claim and
    are excluded from both numerator and denominator.  Including them
    would penalize honest abstentions (which cite nothing by design) and
    reward vacuous conclusions.

3.  **Trivial-claim floor.**  A "step" with a claim shorter than 3 words
    is not a rationale claim -- it's a schema filler.  Excluded from the
    denominator.

4.  **Shared resolver.**  MFS reuses `src.teacher.evidence_resolution`
    exactly as the QC G2 gate does.  This is load-bearing: if MFS and G2
    disagreed on what "resolves", our Phase E results would contradict
    our Phase B QC reports.

5.  **Fractional mode.**  Alongside the strict binary score, we expose a
    `mode="fractional"` variant where each step is scored as
    (#resolved / #cited) rather than 1-if-any.  This is useful for
    diagnosing whether teachers cite *many* evidence ids per step (good)
    versus one token per step (brittle).

Aggregation
-----------
- `mfs_trace(trace, context_ids)` -> float in [0, 1], the per-trace score.
- `mfs_corpus(records, ...)`      -> dict with macro (family-balanced) and
                                     weighted (simple-mean-over-traces)
                                     aggregates + per-family breakdown.

A "record" for corpus aggregation is
    { "trace": <dict in rubric schema>,
      "context_ids": <iterable[str]>,
      "family": <str>,                                 # optional
      "trace_id": <str> }                              # optional

Typical usage
-------------
    from src.metrics.mfs import mfs_trace, mfs_corpus
    score = mfs_trace(trace_dict, context_ids_set)

    report = mfs_corpus(eval_records)
    print(report["macro_mfs"], report["per_family_mfs"])
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from src.teacher.evidence_resolution import expand_context_ids, resolves


# Roles that are meta / terminal and therefore not rationale claims.
# (Mirrors schema.py's VALID_ROLES; kept local so MFS can be imported
# without pulling in the full schema module.)
_META_ROLES = {"conclusion", "abstention", "evidence_gap"}
_MIN_CLAIM_WORDS = 3


@dataclass(frozen=True)
class StepMfs:
    """Per-step MFS diagnostics (for report-drilling)."""
    step_id: int
    role: str | None
    n_cited: int
    n_resolved: int
    counted: bool      # True iff this step contributes to the trace score
    score: float       # 0.0 or 1.0 in binary mode; [0,1] in fractional mode


def _is_rationale_step(step: dict) -> bool:
    """A step counts as a rationale claim (included in MFS denominator) iff
    it has a role outside the meta set AND a non-trivial claim text.
    """
    if not isinstance(step, dict):
        return False
    role = step.get("role")
    if role in _META_ROLES:
        return False
    claim = step.get("claim", "") or ""
    if not isinstance(claim, str):
        return False
    return len(claim.split()) >= _MIN_CLAIM_WORDS


def mfs_step(step: dict, expanded: set[str],
             mode: str = "binary") -> StepMfs:
    """Score a single step.

    Args:
        step:       parsed step dict {step_id, role, claim, evidence_ids, ...}
        expanded:   pre-expanded context id pool
                    (from `expand_context_ids(context_ids)`)
        mode:       "binary"      -> 1 if any evidence_id resolves else 0
                    "fractional"  -> (# resolved) / (# cited)

    Returns:
        StepMfs record.
    """
    if mode not in {"binary", "fractional"}:
        raise ValueError(f"mfs_step: unknown mode {mode!r}")

    counted = _is_rationale_step(step)
    step_id = int(step.get("step_id", -1))
    role = step.get("role")

    eids = step.get("evidence_ids") or []
    if not isinstance(eids, list):
        eids = []

    resolved = [e for e in eids if resolves(e, expanded)]
    n_cited = len(eids)
    n_res = len(resolved)

    if not counted:
        return StepMfs(step_id, role, n_cited, n_res, counted=False, score=0.0)

    if mode == "binary":
        score = 1.0 if n_res > 0 else 0.0
    else:  # fractional
        score = (n_res / n_cited) if n_cited > 0 else 0.0

    return StepMfs(step_id, role, n_cited, n_res, counted=True, score=score)


def mfs_trace(trace: dict, context_ids: Iterable[str],
              mode: str = "binary") -> float:
    """Macro-over-steps MFS for one trace.

    Returns 0.0 if the trace has no rationale steps (degenerate: all steps
    are meta-only).  A conservative convention -- such traces don't carry
    any mechanism-faithfulness signal so they shouldn't inflate the score.
    """
    if not isinstance(trace, dict):
        return 0.0
    steps = trace.get("steps") or []
    if not isinstance(steps, list):
        return 0.0

    expanded = expand_context_ids(context_ids)
    scores = []
    for s in steps:
        sm = mfs_step(s, expanded, mode=mode)
        if sm.counted:
            scores.append(sm.score)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def mfs_trace_detailed(trace: dict, context_ids: Iterable[str],
                       mode: str = "binary") -> dict:
    """Trace-level MFS + per-step breakdown (for audits and figures)."""
    expanded = expand_context_ids(context_ids)
    steps = trace.get("steps") or [] if isinstance(trace, dict) else []
    per_step = [mfs_step(s, expanded, mode=mode) for s in steps if isinstance(s, dict)]
    counted = [sm.score for sm in per_step if sm.counted]
    trace_score = (sum(counted) / len(counted)) if counted else 0.0
    return {
        "trace_mfs": trace_score,
        "n_rationale_steps": len(counted),
        "n_meta_steps": sum(1 for sm in per_step if not sm.counted),
        "n_total_cited": sum(sm.n_cited for sm in per_step),
        "n_total_resolved": sum(sm.n_resolved for sm in per_step),
        "per_step": per_step,
        "mode": mode,
    }


def mfs_corpus(records: Sequence[dict], mode: str = "binary",
               default_family: str = "UNKNOWN") -> dict:
    """Aggregate MFS over a corpus of (trace, context_ids, family?) records.

    Each record is expected to contain:
        - "trace":        parsed trace dict (with 'steps')
        - "context_ids":  iterable of str (the retrieval bundle's ids)
        - "family":       optional stratifier (recommended for paper tables)

    Returns:
        {
            "macro_mfs":       float,   # avg over families (stratum-balanced)
            "weighted_mfs":    float,   # avg over traces (simple mean)
            "per_family_mfs":  dict[str, float],
            "per_family_n":    dict[str, int],
            "n_traces":        int,
            "n_degenerate":    int,     # traces with zero rationale steps
            "mode":            str,
        }
    """
    per_trace: list[tuple[str, float, int]] = []  # (family, trace_mfs, n_counted)
    for rec in records:
        tr = rec.get("trace") if isinstance(rec, dict) else None
        if not isinstance(tr, dict):
            continue
        ctx = rec.get("context_ids") or set()
        fam = rec.get("family") or default_family

        detailed = mfs_trace_detailed(tr, ctx, mode=mode)
        per_trace.append((fam, detailed["trace_mfs"], detailed["n_rationale_steps"]))

    n_traces = len(per_trace)
    n_degenerate = sum(1 for _, _, k in per_trace if k == 0)

    by_fam: dict[str, list[float]] = defaultdict(list)
    for fam, s, k in per_trace:
        by_fam[fam].append(s)

    per_family = {f: (sum(v) / len(v) if v else 0.0) for f, v in by_fam.items()}
    per_family_n = {f: len(v) for f, v in by_fam.items()}

    macro = (sum(per_family.values()) / len(per_family)) if per_family else 0.0
    weighted = (sum(s for _, s, _ in per_trace) / n_traces) if n_traces else 0.0

    return {
        "macro_mfs":      macro,
        "weighted_mfs":   weighted,
        "per_family_mfs": per_family,
        "per_family_n":   per_family_n,
        "n_traces":       n_traces,
        "n_degenerate":   n_degenerate,
        "mode":           mode,
    }
