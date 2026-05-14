r"""HR -- Hallucination Rate (novelty pillar P5, also drives G6 in QC).

Definition (plan §8, §11)
-------------------------
    HR = fraction of rationale ENTITY MENTIONS that are not found in the
         known DrugBank / KEGG / enzyme vocabulary.

Motivation
----------
MFS asks: are the CITED evidence IDs real? (resolvable against the
retrieval bundle)  HR asks a broader question: are the ENTITIES mentioned
in the rationale prose real?  Models sometimes invent:
  - enzyme names ("CYP22Q4")
  - fictional DrugBank IDs ("DB999999")
  - made-up UniProt ids / pathway ids / transporter families

HR catches these even when they aren't cited as evidence.  The plan
§11 target is HR <= 0.10.

Scope
-----
HR is designed to work with ANY trace (teacher, student, baseline) and
with ANY reference entity vocabulary.  The caller supplies:
  - A regex pipeline for EXTRACTING candidate entity mentions from
    step claims (default covers enzyme names, DrugBank IDs, UniProt
    IDs, pathway IDs, and known transporter families).
  - A set of KNOWN entities (derived from DrugBank + KEGG + the
    canonical enzyme list).
We compare extracted candidates against the known set and count the
unknowns as hallucinations.

Design choices
--------------
1.  **Case-insensitive lookups.**  `CYP3A4` and `cyp3a4` should both
    count as known if the canonical form is present in the vocabulary.

2.  **Normalization via `evidence_resolution`.**  We reuse
    `normalize_citation` / `expand_context_ids` so HR's resolution
    rules match MFS / G2.  That is load-bearing: if HR flagged entities
    that MFS accepted (or vice versa), Phase E numbers would
    contradict each other.

3.  **Default extractor is conservative.**  It only extracts high-
    confidence entity-like tokens (CYP + digits + letter + digits,
    OATP*, P-gp, DB + digits, UniProt-pattern, hsa: + digits, etc.)
    It deliberately does NOT extract arbitrary capitalized words
    (too noisy, would flag common English words as "entities").

4.  **Report granularity.**  Per-trace HR + per-family HR + an
    unknowns-histogram for drilling into WHICH fake entities the model
    is most likely to invent.

Usage
-----
    from src.metrics.hr import hr_corpus, default_entity_extractor, load_known_entities
    known = load_known_entities()       # from DrugBank parquets
    rep = hr_corpus(records, known_entities=known)
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from src.teacher.evidence_resolution import expand_context_ids

EntityExtractor = Callable[[str], list[str]]

_META_ROLES = {"conclusion", "abstention", "evidence_gap"}


# ----------------------------------------------------------------------
# Default entity extractor.
# ----------------------------------------------------------------------
# Matches common DDI-relevant entities.  Each pattern is designed to fire
# only on tokens that look like entities -- avoiding common English false
# positives.  Order matters (longer patterns first -- CYP3A4 before CYP).
# ----------------------------------------------------------------------
_ENTITY_PATTERNS = [
    # DrugBank-like IDs
    (re.compile(r"\bDB\d{5,}\b"),                           "drugbank_id"),
    # CYP enzymes: CYP3A4, CYP2D6, CYP11A1, and hallucinations like CYP22Q4.
    # Pattern: CYP + 1-2 digits + optional letter + 0-2 digits.  Permissive
    # by design -- HR's job is to FLAG ids like CYP22Q4 as unknown, so we
    # must extract them.
    (re.compile(r"\bCYP\d{1,2}[A-Z]\d{0,2}\b", re.IGNORECASE), "cyp_enzyme"),
    # OATP family (OATP1B1, OATP2B1)
    (re.compile(r"\bOATP\d[A-Z]\d{0,2}\b", re.IGNORECASE),  "oatp"),
    # Generic P-gp / Pgp / ABCB1
    (re.compile(r"\b(?:P[- ]?gp|PGP|ABCB1)\b", re.IGNORECASE), "pgp"),
    # Other efflux / uptake families
    (re.compile(r"\b(?:BCRP|MRP\d|MATE\d?|OCT\d|OAT\d|NTCP|BSEP)\b", re.IGNORECASE),
                                                             "transporter"),
    # KEGG pathway IDs  (hsa:12345  /  hsa05200)
    (re.compile(r"\bhsa[:\-]?\d{4,6}\b", re.IGNORECASE),     "kegg_pathway"),
    # UniProt IDs (6- or 10-char alnum, letter-start)
    (re.compile(r"\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b"),          "uniprot"),
    (re.compile(r"\b[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9](?:[A-Z][A-Z0-9]{2}[0-9])?\b"), "uniprot"),
]


def default_entity_extractor(text: str) -> list[str]:
    """Return list of candidate entity strings mentioned in `text`.

    Conservative -- only fires on patterns that look like biomedical
    entities.  Returns the matched substrings as-is (no normalization)
    so the caller can report exact hallucinations.
    """
    if not isinstance(text, str) or not text:
        return []
    out: list[str] = []
    for rx, _kind in _ENTITY_PATTERNS:
        out.extend(rx.findall(text))
    return out


def _is_known(entity: str, known_expanded: set[str]) -> bool:
    """Case-insensitive / alias-aware membership check.

    Reuses the same resolver as MFS / G2.  `known_expanded` should be
    the output of `expand_context_ids(known_entities)`.
    """
    if not isinstance(entity, str) or not entity:
        return False
    candidates = {entity, entity.lower(), entity.upper()}
    # Also strip hyphens and spaces for family names like "P-gp" / "P gp"
    compact = re.sub(r"[\s\-_]", "", entity.lower())
    candidates.add(compact)
    return any(c in known_expanded for c in candidates)


@dataclass(frozen=True)
class HrTrace:
    trace_id: str
    n_entities: int
    n_unknown: int
    unknown_entities: list[str]
    hr: float      # per-trace HR in [0, 1]
    gold_family: str | None = None


def hr_trace(trace: dict,
             known_entities: Iterable[str],
             extractor: EntityExtractor | None = None,
             trace_id: str | None = None,
             gold_family: str | None = None,
             include_meta_steps: bool = False) -> HrTrace:
    """Per-trace HR.

    known_entities: vocabulary of real entities (drug names, enzyme ids,
                     DrugBank ids, UniProt ids, pathway ids, ...).
    extractor:      optional override for candidate-extraction logic.
    include_meta_steps: if False (default), skip conclusion/abstention/
                     evidence_gap steps -- the typical rationale claims
                     are where hallucinations live.
    """
    if extractor is None:
        extractor = default_entity_extractor
    known_expanded = expand_context_ids(known_entities)

    all_ents: list[str] = []
    unknowns: list[str] = []

    steps = (trace.get("steps") or []) if isinstance(trace, dict) else []
    for s in steps:
        if not isinstance(s, dict):
            continue
        if (not include_meta_steps) and s.get("role") in _META_ROLES:
            continue
        claim = s.get("claim", "") or ""
        for ent in extractor(claim):
            all_ents.append(ent)
            if not _is_known(ent, known_expanded):
                unknowns.append(ent)

    n_ent = len(all_ents)
    hr = (len(unknowns) / n_ent) if n_ent else 0.0
    return HrTrace(
        trace_id=trace_id or trace.get("trace_id", "") if isinstance(trace, dict) else "",
        n_entities=n_ent,
        n_unknown=len(unknowns),
        unknown_entities=unknowns,
        hr=hr,
        gold_family=gold_family,
    )


def hr_corpus(records: Sequence[dict],
              known_entities: Iterable[str],
              extractor: EntityExtractor | None = None,
              include_meta_steps: bool = False) -> dict:
    """Aggregate HR over a corpus.

    Records:  [{ "trace": <dict>, "gold_family": <str>?, "trace_id": <str>? }, ...]

    Returns:
        {
            "hr":             float,   # macro over families
            "weighted_hr":    float,   # simple mean over traces
            "micro_hr":       float,   # (sum unknown) / (sum entities)
            "per_family_hr":  dict,
            "unknown_top":    list[(entity, count)],   # most common hallucinations
            "n_traces":       int,
            "n_entities":     int,
            "n_unknown":      int,
        }
    """
    if extractor is None:
        extractor = default_entity_extractor
    known_expanded = expand_context_ids(known_entities)

    per_trace: list[tuple[str, float, int, int]] = []
    unknowns_counter: Counter[str] = Counter()
    total_ent = 0
    total_unk = 0

    for rec in records:
        tr = rec.get("trace") if isinstance(rec, dict) else None
        if not isinstance(tr, dict):
            continue
        fam = rec.get("gold_family") or "UNKNOWN"
        # Inline traversal to avoid re-expanding known_expanded per trace
        n_ent = 0
        n_unk = 0
        steps = tr.get("steps") or []
        for s in steps:
            if not isinstance(s, dict):
                continue
            if (not include_meta_steps) and s.get("role") in _META_ROLES:
                continue
            claim = s.get("claim", "") or ""
            for ent in extractor(claim):
                n_ent += 1
                if not _is_known(ent, known_expanded):
                    n_unk += 1
                    unknowns_counter[ent] += 1
        trace_hr = (n_unk / n_ent) if n_ent else 0.0
        per_trace.append((fam, trace_hr, n_ent, n_unk))
        total_ent += n_ent
        total_unk += n_unk

    n_traces = len(per_trace)
    weighted = (sum(h for _, h, _, _ in per_trace) / n_traces) if n_traces else 0.0
    micro = (total_unk / total_ent) if total_ent else 0.0

    by_fam: dict[str, list[float]] = defaultdict(list)
    for fam, h, _, _ in per_trace:
        by_fam[fam].append(h)
    per_family = {f: (sum(v) / len(v) if v else 0.0) for f, v in by_fam.items()}
    macro = (sum(per_family.values()) / len(per_family)) if per_family else 0.0

    return {
        "hr":             macro,
        "weighted_hr":    weighted,
        "micro_hr":       micro,
        "per_family_hr":  per_family,
        "unknown_top":    unknowns_counter.most_common(20),
        "n_traces":       n_traces,
        "n_entities":     total_ent,
        "n_unknown":      total_unk,
    }


# ----------------------------------------------------------------------
# Helper: load a default known-entity vocabulary from the V4 parquets.
# ----------------------------------------------------------------------
def load_known_entities(data_processed: Path | str | None = None) -> set[str]:
    """Load a default DrugBank / KEGG / enzyme vocabulary.

    Assembles:
      - DrugBank IDs and drug names from `drugs.parquet`
      - DrugBank-known UniProt IDs from `drug_proteins.parquet`
      - KEGG pathway IDs from `pathways_unified.parquet`
      - A hard-coded list of canonical CYP / transporter families
        (since these are universal, not V4-specific)

    Intended for offline metric reporting; not imported at metric
    compute time so callers can override.
    """
    import pyarrow.parquet as pq  # local import -- only needed for CLI use

    if data_processed is None:
        data_processed = Path(__file__).resolve().parents[2] / "data_processed"
    data_processed = Path(data_processed)

    out: set[str] = set()

    # Canonical CYP + transporter vocabulary (always valid).
    out.update([
        "CYP1A2", "CYP2A6", "CYP2B6", "CYP2C8", "CYP2C9", "CYP2C19",
        "CYP2D6", "CYP2E1", "CYP3A4", "CYP3A5",
        "P-gp", "Pgp", "ABCB1", "BCRP", "ABCG2",
        "OATP1B1", "OATP1B3", "OATP2B1", "OCT1", "OCT2",
        "OAT1", "OAT3", "MATE1", "MATE2", "MRP2", "MRP4",
        "NTCP", "BSEP",
    ])

    for parquet_file, cols in [
        ("drugs.parquet",          ["drugbank_id", "name"]),
        ("drug_proteins.parquet",  ["uniprot"]),
        ("pathways_unified.parquet", ["pathway_id"]),
    ]:
        p = data_processed / parquet_file
        if not p.exists():
            continue
        try:
            tbl = pq.read_table(p, columns=[c for c in cols])
            for col in cols:
                if col in tbl.column_names:
                    for v in tbl.column(col).to_pylist():
                        if isinstance(v, str) and v:
                            out.add(v)
        except Exception:
            continue

    return out
