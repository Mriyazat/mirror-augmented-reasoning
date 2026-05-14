"""Evidence-ID resolution shared between the QC gate G2 and the Phase E
metrics (MFS, HR, RIS).

Rationale
---------
Teacher traces cite evidence IDs like `cyp3a4_sub`, `pathway_jaccard`, or
`hsa-mmu:05200`.  Those IDs are drawn from a `context_ids()` set built by
`ContextBuilder`.  Models drift from the canonical forms in three ways:

  1. PK-flag suffix style.   We emit `cyp3a4_sub`; models prefer
                              `CYP3A4_substrate`, `Cyp3A4-inhibitor`, etc.
  2. Bare enzyme name.       If the pool contains `cyp2d6_sub`, the model
                              may cite just `CYP2D6` or `CYP 2D6`.
  3. Metric name + value.    Context is `pathway_jaccard`; model emits
                              `pathway_jaccard = 0.000` as one string.

Both the QC G2 gate (binary hallucination check) and MFS (continuous
faithfulness score) need to agree on the resolution rule -- if the rule
drifts between the two, the metric disagrees with the gate on the same
trace.  Keeping the helpers in a single module is the canonical fix.

Public API
----------
    normalize_citation(tok)       -> str
    expand_context_ids(ids)       -> set[str]
    strip_metric_literal(tok)     -> str
    resolves(eid, expanded_pool)  -> bool       (evidence lookup used by callers)

The implementation is the V4 post-patch version (April 2026) that handles
Qwen's `_substrate`/`_inducer` drift, DeepSeek's bare enzyme names, and
multi-teacher metric-literal citations.
"""
from __future__ import annotations

import re
from typing import Iterable


# PK-flag suffix normalization. Qwen emits `_substrate` / `_inducer`
# instead of our canonical `_sub` / `_ind`, with mixed case.
_PK_SUFFIX_ALIASES = {
    "substrate": "sub", "substrates": "sub",
    "inhibitor": "inh", "inhibitors": "inh", "inhibit": "inh",
    "inducer": "ind", "inducers": "ind", "induce": "ind",
    "sub": "sub", "inh": "inh", "ind": "ind",
}

# Matches <enzyme>_<suffix> with any suffix and any casing.
_FLAG_RX = re.compile(
    r"^([a-z][a-z0-9]*)[_-](sub(?:strate|strates)?|inh(?:ibitor|ibitors|ibit)?|ind(?:ucer|ucers|uce)?)$",
    re.IGNORECASE,
)

# Transporter / enzyme bare-name families auto-whitelisted when any
# `<name>_<suffix>` form is present in the context.
_ENZYME_FAMILY_RX = re.compile(
    r"^(cyp[0-9a-z]+|p[_-]?gp|oatp[0-9a-z]*|bcrp|oct[0-9]?|mrp[0-9]?|"
    r"mate[0-9]?|oat[0-9]?|ntcp|bsep|mdr[0-9]?|abc[a-z][0-9]*)$",
    re.IGNORECASE,
)


def normalize_citation(tok: str) -> str:
    """Lowercase + normalize PK-flag suffix.

    Example:
        "CYP3A4_substrate" -> "cyp3a4_sub"
        "P-gp_inhibitor"   -> "p_gp_inh"
        "pathway_jaccard"  -> "pathway_jaccard"      (unchanged)
    """
    if not isinstance(tok, str):
        return tok  # type: ignore[return-value]
    t = tok.strip().lower().replace("-", "_")
    m = _FLAG_RX.match(t)
    if m:
        base, suffix = m.group(1), m.group(2).lower()
        canon = _PK_SUFFIX_ALIASES.get(suffix, suffix[:3])
        return f"{base}_{canon}"
    return t


def strip_metric_literal(tok: str) -> str:
    """`pathway_jaccard = 0.000` -> `pathway_jaccard`.

    Models cite the whole `name = value` line as one string even though
    only the metric name is in the context pool.
    """
    if not isinstance(tok, str):
        return tok  # type: ignore[return-value]
    return re.split(r"\s*=\s*", tok, maxsplit=1)[0].strip()


def expand_context_ids(context_ids: Iterable[str]) -> set[str]:
    """Enrich the evidence pool with the aliases models reasonably use.

    Covers:
      - case variants of every id
      - PK-flag suffix variants (`_substrate` -> `_sub`, `_inducer` -> `_ind`)
      - bare enzyme / transporter names implied by matching `_sub/_inh/_ind`
        flags (`cyp2d6_sub` in pool -> accept `cyp2d6`, `CYP2D6`, `CYP 2D6`)
      - P-gp / PGP / ABCB1 synonyms of `pgp`

    Does NOT add arbitrary substring matches -- we still reject invented
    DrugBank / UniProt IDs or uncited enzymes that only appear in the
    model's rationale.
    """
    expanded: set[str] = set()
    enzyme_bases: set[str] = set()
    for cid in context_ids:
        if not isinstance(cid, str) or not cid:
            continue
        expanded.add(cid)
        expanded.add(cid.lower())
        expanded.add(cid.upper())
        expanded.add(normalize_citation(cid))
        m = _FLAG_RX.match(cid.lower())
        if m:
            enzyme_bases.add(m.group(1).lower())
    for base in enzyme_bases:
        if not _ENZYME_FAMILY_RX.match(base):
            continue
        expanded.add(base)
        expanded.add(base.upper())
        # P-gp family aliases
        if base.replace("-", "").replace("_", "") == "pgp":
            expanded.update({"P-gp", "p-gp", "PGP", "pgp", "ABCB1", "abcb1"})
    expanded.discard("")
    expanded.discard(None)  # type: ignore[arg-type]
    return expanded


def resolves(eid: str, expanded: set[str]) -> bool:
    """True iff `eid` matches any canonical form in the expanded context pool.

    Tries five forms: as-is, lowercased, metric-stripped (both cases),
    PK-flag-normalized.  Mirrors exactly the candidate-set logic used by
    the QC G2 gate (see `src.teacher.qc._gate_evidence`).
    """
    if not isinstance(eid, str):
        return False
    candidates = {
        eid,
        eid.lower(),
        strip_metric_literal(eid),
        strip_metric_literal(eid).lower(),
        normalize_citation(eid),
    }
    return bool(candidates & expanded)
