"""RPC -- Reasoning-Prediction Coherence.
Definition
----------
    RPC = 1 - P(rationale-implied label != final prediction)

Motivation
----------
A "silence rewards precision" failure mode observed in earlier student
baselines manifested as a model that hedged through the rationale
("I don't have enough evidence", "could be metabolism or adverse risk")
and then committed to an arbitrary family in `final_answer.family`. The
rationale implied one thing while the final answer said another. RPC
catches this by asking: what family does the REASONING point to, and
does it match what the model FINALLY predicted?

The challenge: how do we extract the "rationale-implied label"?
---------------------------------------------------------------
We need a label classifier that:
  (i)  operates on the trace steps (NOT on `final_answer`) so we can
       compare the two.
  (ii) is cheap and deterministic (so RPC can be run on every trace).
  (iii) is pluggable (so the LLM-as-judge variant in evaluation can drop
        in without rewriting the metric).

Default classifier: lexical family-cue matching on step claims.
We scan each rationale step's `claim` text for keywords associated with
each family in the taxonomy and return argmax. This is the cheapest
defensible extractor; for paper-quality RPC the lexical extractor should
be cross-validated with an LLM-as-judge on a small held-out subset.

The lexical classifier deliberately does NOT read `final_answer`, so
RPC is a meaningful number -- the rationale's implied label is computed
independently of the commitment.

Abstention handling
-------------------
An abstaining trace has `final_answer.abstain = True`.  RPC for such a
trace is 1 iff the rationale contains an `abstention` or `evidence_gap`
role step (the reasoning also abstained).  Otherwise RPC = 0: the model
abstained without the reasoning supporting that abstention, which is the
"silence rewards precision" failure mode.

Classifier signature (pluggable)
--------------------------------
    Classifier = Callable[[dict], str | None]
        Takes a trace dict, returns the implied family label or None
        if the rationale is indeterminate (no cues match).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

Classifier = Callable[[dict], str | None]


# ------------------------------------------------------------------
# Lexical family-cue dictionary (the family taxonomy)
#
# Taxonomy (from data_processed/taxonomy_schema.json):
#   AdverseRisk  Efficacy  Other  PD_Activity  PK_Absorption
#   PK_Distribution  PK_Excretion  PK_Metabolism
#
# Keywords are chosen to be HIGH-PRECISION: a single match is strong
# evidence for the family, and families don't collide on shared cues
# where possible. Inevitable overlap cases (e.g. "QT prolongation"
# appears in both AdverseRisk and PD_Activity) are resolved by the
# count + first-seen tie-breaker in `lexical_family_classifier`.
#
# Revised cue dictionary:
# (previously missing -> all traces about these families scored as
# `indeterminate` and were counted as incoherent on commit). Removed
# stale cues for PK_Transport / PD_Synergy / PD_Antagonism /
# Contraindication / Immunological (not families in the current taxonomy).
# ------------------------------------------------------------------
_FAMILY_CUES: dict[str, list[str]] = {
    "PK_Metabolism": [
        r"\bmetaboli[sz]e?d?\b", r"\bmetabolism\b",
        r"\bcyp\s?\d", r"\bcyp[0-9]{1}[a-z]",
        r"\benzyme\b", r"\bp[- ]?450\b",
        r"\binduce.{0,20}(metaboli|cyp|enzyme)",
        r"\binhibit.{0,20}(metaboli|cyp|enzyme)",
        r"\bhepatic clearance\b", r"\bdemethylation\b",
        r"\boxidation\b",
    ],
    "PK_Absorption": [
        r"\babsorption\b", r"\babsorb\b",
        r"\bchelat", r"\bcomplex formation\b",
        r"\bgastric (emptying|ph)\b", r"\bgi absorption\b",
        r"\bbioavailability\b",
    ],
    "PK_Distribution": [
        r"\bprotein[- ]?bind", r"\bdisplace.{0,20}(protein|binding|albumin)",
        r"\balbumin\b", r"\bplasma (protein|bound)\b",
        r"\bdistribution volume\b", r"\bserum concentration\b",
        # Transporters often modulate distribution (BBB, P-gp, OATP)
        r"\bp[- ]?gp\b", r"\babcb1\b", r"\boatp",
    ],
    "PK_Excretion": [
        r"\brenal clearance\b", r"\bexcret",
        r"\belimination half[- ]?life\b",
        r"\btubular secretion\b", r"\bglomerular\b",
        r"\burinary excretion\b", r"\bbiliary\b",
        r"\brenal tubule\b",
    ],
    "PD_Activity": [
        r"\bpharmacodynamic\b",
        r"\badditive (effect|activity|action)\b",
        r"\bsynerg", r"\bpotentia[lt]e\b",
        r"\bboth (drugs )?(increase|decrease|enhance|reduce|share)\b",
        r"\bshare.{0,20}(mechanism|activity|pharmacolog)",
        r"\benhance the (activity|effect|action)\b",
        r"\bantagoniz(e|es|ed|ing)\b",
        r"\boppos(e|ing|es) the (action|effect|activity)\b",
        r"\breceptor (agonist|antagonist|activity)\b",
        # High-precision activity-subtype mentions
        r"\banti[- ]?hypertensive\b", r"\bhypotensive\b",
        r"\bcns depress", r"\bsedative\b",
        r"\banticoagulant\b", r"\bantiplatelet\b",
        r"\bhypoglycemic effect\b", r"\bvasodilat",
        r"\bserotonergic\b", r"\banticholinergic\b",
    ],
    "Efficacy": [
        r"\btherapeutic efficacy\b", r"\btherapeutic effect\b",
        r"\b(reduc|decreas|diminish|impair|lower).{0,20}(efficacy|effectiveness)\b",
        r"\bloss of (efficacy|effectiveness)\b",
        r"\bdiagnostic (effectiveness|efficacy|purpose|utility)\b",
        r"\btreatment failure\b", r"\bfailure of therapy\b",
    ],
    "AdverseRisk": [
        # Explicit risk / causation language
        r"\b(increase|elevat|heighten|raise).{0,20}risk of\b",
        r"\brisk of\b",
        r"\bmay (cause|lead to|result in)\b",
        # Adverse-outcome lexicon (symptoms / toxicities)
        r"\bqt prolong", r"\btorsade",
        r"\bbleed", r"\bhemorrhag", r"\bthrombocytopen",
        r"\bhyperkalem", r"\bhypokalem",
        r"\bhypoglycem(?!ic effect)",  # "hypoglycem*" but NOT "hypoglycemic effect" (PD_Activity)
        r"\bhypertension\b", r"\bhypotension\b",
        r"\bserotonin syndrome\b",
        r"\bnephrotox", r"\bhepatotox", r"\bcardiotox", r"\bneurotox", r"\bototox",
        r"\bmyopathy\b", r"\brhabdomyol", r"\bseizure\b",
        r"\bmethemoglobinemia\b", r"\bangioedema\b",
    ],
    "Other": [
        r"\bunmatched\b", r"\bmiscellaneous\b",
        r"\buncategoriz(ed|able)\b",
    ],
}

_FAMILY_CUE_REGEX = {
    fam: [re.compile(p, re.IGNORECASE) for p in cues]
    for fam, cues in _FAMILY_CUES.items()
}


# Step roles excluded from the rationale extraction -- meta / terminal steps
# don't reason about family, they summarize or abstain.
_META_ROLES = {"conclusion", "abstention", "evidence_gap"}


def lexical_family_classifier(trace: dict) -> str | None:
    """Default rationale-implied-family extractor.

    Scans rationale-step claims for family-cue regex patterns and
    returns the argmax family.  Returns None if no cues match.

    Ties are broken by FIRST-APPEARANCE in the steps (earlier-reasoned
    family wins) because in clinical notes the first mechanism
    mentioned is usually the primary one.
    """
    if not isinstance(trace, dict):
        return None
    steps = trace.get("steps") or []
    if not isinstance(steps, list):
        return None

    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if step.get("role") in _META_ROLES:
            continue
        claim = step.get("claim", "") or ""
        if not isinstance(claim, str):
            continue
        for fam, rxs in _FAMILY_CUE_REGEX.items():
            for rx in rxs:
                if rx.search(claim):
                    counts[fam] += 1
                    first_seen.setdefault(fam, idx)
                    break  # 1 hit per family per step

    if not counts:
        return None
    max_count = max(counts.values())
    tied = [f for f, c in counts.items() if c == max_count]
    if len(tied) == 1:
        return tied[0]
    # break ties by first-seen index
    tied.sort(key=lambda f: first_seen[f])
    return tied[0]


def _rationale_supports_abstention(trace: dict) -> bool:
    """True iff any rationale step has role in {abstention, evidence_gap}."""
    if not isinstance(trace, dict):
        return False
    steps = trace.get("steps") or []
    return any(isinstance(s, dict) and s.get("role") in {"abstention", "evidence_gap"}
               for s in steps)


@dataclass(frozen=True)
class RpcRecord:
    """Per-trace RPC evaluation."""
    trace_id: str
    implied_label: str | None
    final_label: str | None
    final_abstained: bool
    rationale_supports_abstention: bool
    is_coherent: bool
    gold_family: str | None = None


def rpc_trace(trace: dict,
              classifier: Classifier | None = None,
              trace_id: str | None = None,
              gold_family: str | None = None) -> RpcRecord:
    """Score coherence on one trace."""
    if classifier is None:
        classifier = lexical_family_classifier

    final = trace.get("final_answer", {}) if isinstance(trace, dict) else {}
    final_label = final.get("family")
    final_abstained = bool(final.get("abstain", False))

    supports_abstain = _rationale_supports_abstention(trace)
    implied = classifier(trace)

    if final_abstained:
        # Abstention is coherent iff reasoning also abstained.
        coherent = supports_abstain
    else:
        # Commitment is coherent iff the rationale points at the same family.
        # `implied is None` means the rationale didn't make a case for ANY
        # family -- we call that INCOHERENT with a commitment: the model
        # committed without evidence-anchored reasoning.
        coherent = (implied is not None) and (implied == final_label)

    return RpcRecord(
        trace_id=trace_id or trace.get("trace_id", ""),
        implied_label=implied,
        final_label=final_label,
        final_abstained=final_abstained,
        rationale_supports_abstention=supports_abstain,
        is_coherent=coherent,
        gold_family=gold_family,
    )


def rpc_corpus(records: Iterable[dict],
               classifier: Classifier | None = None) -> dict:
    """Aggregate RPC over a corpus.  Each record is a dict with:
        "trace":       parsed trace dict
        "trace_id":    str  (optional)
        "gold_family": str  (optional, for stratified report)

    Returns a report with overall RPC, per-family breakdown, and
    diagnostic counts (how often the classifier was indeterminate,
    how often abstention-coherence was the route, etc.).
    """
    rows: list[RpcRecord] = []
    for rec in records:
        tr = rec.get("trace") if isinstance(rec, dict) else None
        if not isinstance(tr, dict):
            continue
        rows.append(rpc_trace(
            tr, classifier=classifier,
            trace_id=rec.get("trace_id"),
            gold_family=rec.get("gold_family"),
        ))

    n = len(rows)
    n_coherent = sum(1 for r in rows if r.is_coherent)
    n_abstain = sum(1 for r in rows if r.final_abstained)
    n_abstain_coherent = sum(1 for r in rows if r.final_abstained and r.is_coherent)
    n_committed = n - n_abstain
    n_committed_coherent = n_coherent - n_abstain_coherent
    n_indeterminate = sum(1 for r in rows if (not r.final_abstained) and r.implied_label is None)

    by_fam_total: dict[str, int] = defaultdict(int)
    by_fam_hit:   dict[str, int] = defaultdict(int)
    for r in rows:
        f = r.gold_family or "UNKNOWN"
        by_fam_total[f] += 1
        if r.is_coherent:
            by_fam_hit[f] += 1

    per_family = {f: (by_fam_hit[f] / by_fam_total[f]) for f in by_fam_total}
    macro_rpc = (sum(per_family.values()) / len(per_family)) if per_family else 0.0

    return {
        "rpc":                         (n_coherent / n) if n else 0.0,
        "macro_rpc":                   macro_rpc,
        "n":                           n,
        "n_coherent":                  n_coherent,
        "n_committed":                 n_committed,
        "n_committed_coherent":        n_committed_coherent,
        "committed_rpc":               (n_committed_coherent / n_committed) if n_committed else 0.0,
        "n_abstain":                   n_abstain,
        "n_abstain_coherent":          n_abstain_coherent,
        "abstain_rpc":                 (n_abstain_coherent / n_abstain) if n_abstain else 0.0,
        "n_indeterminate":             n_indeterminate,
        "indeterminate_rate":          (n_indeterminate / n) if n else 0.0,
        "per_family_rpc":              per_family,
        "per_family_n":                dict(by_fam_total),
    }
