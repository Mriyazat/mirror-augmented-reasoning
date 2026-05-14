"""Schema + rubric loader for DDI teacher traces.

- `Step`, `FinalAnswer`, `Trace`  dataclasses for the teacher output.
- `load_rubric()`  loads configs/prm_rubric.yaml once.
- `validate_step_schema()` / `validate_trace_schema()` schema-parse gate (QC-1).

Dimension-level validation (evidence / direction / family / PK / relevance /
no_silent_abstain) lives in qc.py — this module is pure structural.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # rubric YAML loader will raise if not installed

ROOT = Path(__file__).resolve().parents[2]
RUBRIC_PATH = ROOT / "configs" / "prm_rubric.yaml"

VALID_ROLES = {
    # Evidence-pool inspection roles (each step cites a slice of context).
    "pathway",              # cites shared/per-drug pathway annotations
    "protein",              # cites shared/per-drug protein targets
    "pk_flag",              # cites CYP/P-gp/OATP/BCRP activity flags
    "structural",           # cites SMILES / substructure evidence
    "atc",                  # cites ATC-class overlap
    "mechanism_of_action",  # cites drug-level MoA narrative (DB `moa` field)
    # Retrieval / analogy roles (RAG context).
    "neighbor_pair",        # cites a specific neighbor pair label as analogy
    "pair_similarity",      # cites jaccard/tanimoto/ATC-prefix scores
    # Meta-evidence roles (when evidence is sparse or absent).
    "evidence_gap",         # states what's missing (no shared pathway/protein/etc.)
    "abstention",           # explicit abstention reasoning step
    # Inference roles.
    "direction",            # direction-resolution step
    "conclusion",           # MUST be the last step; ties to final_answer
}
VALID_DIRECTION_TAGS = {"a_to_b", "b_to_a", "bidirectional", "n/a"}
VALID_POLARITIES = {"up", "down", "risk", "risk_down", "risk_up", None}

# Models drift on role names despite the prompt's fixed vocabulary (~2% of
# steps use the synonyms below).  Rather than restart a 4-day run, we
# canonicalize at parse time inside `extract_json_block` — validators and
# downstream consumers only ever see the canonical set.
ROLE_ALIASES: dict[str, str] = {
    # protein variants
    "shared_protein":      "protein",
    "shared_proteins":     "protein",
    "per_drug_protein":    "protein",
    "per_drug_proteins":   "protein",
    "per-drug_protein":    "protein",
    "per-drug_proteins":   "protein",
    "protein_target":      "protein",
    "protein_targets":     "protein",
    "proteins":            "protein",
    "protein_analysis":    "protein",
    # pathway variants
    "shared_pathway":      "pathway",
    "shared_pathways":     "pathway",
    "per_drug_pathway":    "pathway",
    "pathways":            "pathway",
    "pathway_analysis":    "pathway",
    # mechanism variants
    "mechanism":             "mechanism_of_action",
    "mechanism_description": "mechanism_of_action",
    "moa":                   "mechanism_of_action",
    "moa_description":       "mechanism_of_action",
    "drug_moa":              "mechanism_of_action",
    "drug_mechanism":        "mechanism_of_action",
    "mechanism_analysis":    "mechanism_of_action",
    # pk_flag variants
    "pk_flags":              "pk_flag",
    "active_pk_flag":        "pk_flag",
    "active_pk_flags":       "pk_flag",
    # neighbor/analogy variants
    "neighbor":                 "neighbor_pair",
    "neighbor_pairs":           "neighbor_pair",
    "neighbor_analysis":        "neighbor_pair",
    "neighbor_evidence":        "neighbor_pair",
    "mechanistic_neighbor":     "neighbor_pair",
    "mechanistic_neighbors":    "neighbor_pair",
    "mechanistic-neighbor":     "neighbor_pair",
    "analogy":                  "neighbor_pair",
    "analogical_evidence":     "neighbor_pair",
    "analogical_reasoning":    "neighbor_pair",
    "neighbor_analogy":        "neighbor_pair",
    # similarity variants
    "pair_similarity_signatures": "pair_similarity",
    "pair_similarity_analysis":   "pair_similarity",
    "similarity_analysis":        "pair_similarity",
    "pathway_jaccard":            "pair_similarity",
    "protein_jaccard":            "pair_similarity",
    "atc_similarity":             "pair_similarity",
    "atc_prefix_depth":           "pair_similarity",
    "structural_similarity":      "structural",
    "drug_similarity":            "pair_similarity",
    # evidence-gap variants
    "evidence_assessment":   "evidence_gap",
    "evidence_evaluation":   "evidence_gap",
    "evidence_limitation":   "evidence_gap",
    "evidence_inspection":   "evidence_gap",
    "evidence_description":  "evidence_gap",
    "evidence_survey":       "evidence_gap",
    "evidence_summary":      "evidence_gap",
    "evidence_sparse":       "evidence_gap",
    "lack_of_evidence":      "evidence_gap",
    "lack_of_info":          "evidence_gap",
    "lack_of_information":   "evidence_gap",
    "lack_of_overlap":       "evidence_gap",
    "data_sparsity":         "evidence_gap",
    "data_sparseness":       "evidence_gap",
    "sparse_evidence":       "evidence_gap",
    "insufficient_evidence": "evidence_gap",
    "no_overlap":            "evidence_gap",
    "no_shared_pathways":    "evidence_gap",
    "no_shared_proteins":    "evidence_gap",
    "no_pathway_overlap":    "evidence_gap",
    "no_protein_overlap":    "evidence_gap",
    # abstention variants
    "abstention_reasoning":  "abstention",
    "abstention_reason":     "abstention",
    "abstention_decision":   "abstention",
    "abstention_evaluation": "abstention",
    "abstention_hint":       "abstention",
    # conclusion variants
    "conclusion_setup":      "conclusion",
    "conclusion_hint":       "conclusion",
    "conclusion_support":    "conclusion",
    "conclusion_step":       "conclusion",
    "conclusion_helper":     "conclusion",
    "conclusion_prelude":    "conclusion",
    "conclusion_prelim":     "conclusion",
    "conclusion_mechanism":  "conclusion",
    "conclusion_approach":   "conclusion",
}

# "none" / "null" / "" sometimes appear as string polarity when the trace
# means "no polarity" — normalize to Python None.  Qwen also emits "unknown".
_POLARITY_NULL_STRINGS = {"none", "null", "n/a", "", "neutral", "unknown"}

BANNED_ABSTAIN_PHRASES = (
    "i cannot",
    "i can not",
    "insufficient information",
    "not enough information",
    "unable to determine",
    "cannot be determined",
)


# ───────────────────────── dataclasses ─────────────────────────
@dataclass
class Step:
    step_id: int
    claim: str
    evidence_ids: list[str]
    role: str
    direction_tag: str
    polarity: str | None = None
    family_hint: str | None = None


@dataclass
class FinalAnswer:
    family: str
    subtype: str
    direction_tag: str
    polarity: str | None
    confidence: float
    abstain: bool
    summary: str = ""             # ≤80 words soft / 120 hard (rubric summary_constraints)


@dataclass
class Trace:
    pair_id: str
    candidate_id: int             # 0..n_candidates-1
    steps: list[Step]
    final_answer: FinalAnswer
    raw_text: str = ""            # the original generated text (for Med-PRM PRM scoring)
    provider: str = ""            # e.g. "vllm:Llama-3.3-70B-Instruct"
    temperature: float = 0.0
    generation_config: dict = field(default_factory=dict)


# ───────────────────────── rubric loader ─────────────────────────
_RUBRIC_CACHE: dict | None = None


def load_rubric() -> dict:
    """Load configs/prm_rubric.yaml once (cached)."""
    global _RUBRIC_CACHE
    if _RUBRIC_CACHE is not None:
        return _RUBRIC_CACHE
    if yaml is None:
        raise ImportError("pyyaml required: pip install pyyaml")
    _RUBRIC_CACHE = yaml.safe_load(RUBRIC_PATH.read_text())
    return _RUBRIC_CACHE


# ───────────────────────── structural validators ─────────────────────────
@dataclass
class SchemaError:
    where: str
    reason: str


def validate_step_schema(step: dict) -> list[SchemaError]:
    errors: list[SchemaError] = []
    required = ("step_id", "claim", "evidence_ids", "role", "direction_tag")
    for k in required:
        if k not in step:
            errors.append(SchemaError(where=f"step[{step.get('step_id','?')}]",
                                      reason=f"missing required field '{k}'"))
    if errors:
        return errors  # stop early; structure is broken

    if not isinstance(step["step_id"], int) or step["step_id"] < 1:
        errors.append(SchemaError(where=f"step[{step['step_id']}].step_id",
                                  reason="must be int ≥ 1"))
    if not isinstance(step["claim"], str) or not step["claim"].strip():
        errors.append(SchemaError(where=f"step[{step['step_id']}].claim",
                                  reason="must be non-empty string"))
    if not isinstance(step["evidence_ids"], list):
        errors.append(SchemaError(where=f"step[{step['step_id']}].evidence_ids",
                                  reason="must be a list[str]"))
    else:
        for eid in step["evidence_ids"]:
            if not isinstance(eid, str):
                errors.append(SchemaError(where=f"step[{step['step_id']}].evidence_ids",
                                          reason=f"non-string entry: {eid!r}"))
                break
    if step["role"] not in VALID_ROLES:
        errors.append(SchemaError(where=f"step[{step['step_id']}].role",
                                  reason=f"role {step['role']!r} not in {sorted(VALID_ROLES)}"))
    if step["direction_tag"] not in VALID_DIRECTION_TAGS:
        errors.append(SchemaError(where=f"step[{step['step_id']}].direction_tag",
                                  reason=f"direction_tag {step['direction_tag']!r} invalid"))
    if step.get("polarity") not in VALID_POLARITIES:
        errors.append(SchemaError(where=f"step[{step['step_id']}].polarity",
                                  reason=f"polarity {step['polarity']!r} invalid"))
    return errors


def validate_final_answer_schema(ans: dict) -> list[SchemaError]:
    errors: list[SchemaError] = []
    required = ("family", "subtype", "direction_tag", "polarity",
                "confidence", "abstain", "summary")
    for k in required:
        if k not in ans:
            errors.append(SchemaError(where="final_answer",
                                      reason=f"missing required field '{k}'"))
    if errors:
        return errors
    if ans["direction_tag"] not in VALID_DIRECTION_TAGS:
        errors.append(SchemaError(where="final_answer.direction_tag",
                                  reason=f"invalid {ans['direction_tag']!r}"))
    if ans["polarity"] not in VALID_POLARITIES:
        errors.append(SchemaError(where="final_answer.polarity",
                                  reason=f"invalid {ans['polarity']!r}"))
    if not isinstance(ans["confidence"], (int, float)):
        errors.append(SchemaError(where="final_answer.confidence",
                                  reason="must be a number"))
    elif not 0.0 <= float(ans["confidence"]) <= 1.0:
        errors.append(SchemaError(where="final_answer.confidence",
                                  reason="must be in [0, 1]"))
    if not isinstance(ans["abstain"], bool):
        errors.append(SchemaError(where="final_answer.abstain",
                                  reason="must be bool"))
    if not isinstance(ans["summary"], str):
        errors.append(SchemaError(where="final_answer.summary",
                                  reason="must be a string"))
    else:
        # Length validation: soft cap handled by QC G8; here we enforce
        # the absolute hard cap (trace is rejected outright if exceeded).
        wc = len(ans["summary"].split())
        rubric = load_rubric()
        sc = rubric.get("summary_constraints", {})
        soft = int(sc.get("max_words", 80))
        hard = int(sc.get("hard_max_words", soft * 2))
        if wc > hard:
            errors.append(SchemaError(where="final_answer.summary",
                                      reason=f"summary is {wc} words, "
                                      f"exceeds hard cap {hard} "
                                      f"(soft cap {soft})"))
        if not ans["summary"].strip():
            errors.append(SchemaError(where="final_answer.summary",
                                      reason="summary is empty"))
    return errors


def validate_trace_schema(trace: dict) -> list[SchemaError]:
    """Structural gate (QC-1 in rubric). Deeper checks in qc.py."""
    errors: list[SchemaError] = []
    if "steps" not in trace or "final_answer" not in trace:
        errors.append(SchemaError(where="trace", reason="missing 'steps' or 'final_answer'"))
        return errors
    steps = trace["steps"]
    if not isinstance(steps, list) or not steps:
        errors.append(SchemaError(where="trace.steps", reason="must be non-empty list"))
        return errors

    seen_ids: set[int] = set()
    bad_id_types = False
    for s in steps:
        errors.extend(validate_step_schema(s))
        if "step_id" in s:
            sid = s["step_id"]
            if not isinstance(sid, int):
                # Some teachers occasionally emit "1" as a string or even
                # "step_1" -- record the schema error and skip the
                # contiguity check (avoids TypeError on mixed types).
                errors.append(SchemaError(where=f"step[{sid!r}]",
                                          reason=f"step_id must be int, got {type(sid).__name__}"))
                bad_id_types = True
                continue
            if sid in seen_ids:
                errors.append(SchemaError(where=f"step[{sid}]",
                                          reason="duplicate step_id"))
            seen_ids.add(sid)
    # step_ids must be 1..N contiguous (only enforce when all ids were ints)
    if seen_ids and not bad_id_types and (
        min(seen_ids) != 1 or max(seen_ids) != len(steps)
    ):
        errors.append(SchemaError(where="trace.steps",
                                  reason=f"step_ids not 1..{len(steps)} contiguous"))

    # Must contain at least one terminal step: either `conclusion` (commits
    # to a family/direction) or `abstention` (honest uncertainty; teaches
    # the student when NOT to guess).  Both are semantically terminal —
    # they tie to final_answer.  Traces terminating only with intermediate
    # roles (e.g. `pk_flag`, `mechanism_of_action`) are malformed.
    roles_seen = [s.get("role") for s in steps if isinstance(s, dict)]
    if not ({"conclusion", "abstention"} & set(roles_seen)):
        errors.append(SchemaError(
            where="trace.steps",
            reason="no terminal step (need role='conclusion' or 'abstention')",
        ))

    errors.extend(validate_final_answer_schema(trace["final_answer"]))

    # Step length bounds (rubric.teacher_generation)
    rubric = load_rubric()
    tg = rubric["teacher_generation"]
    if len(steps) < tg["min_steps"]:
        errors.append(SchemaError(where="trace.steps",
                                  reason=f"only {len(steps)} steps; min {tg['min_steps']}"))
    if len(steps) > tg["max_steps"]:
        errors.append(SchemaError(where="trace.steps",
                                  reason=f"{len(steps)} steps; max {tg['max_steps']}"))
    return errors


# ───────────────────────── Med-PRM formatter ─────────────────────────
def trace_to_medprm_string(trace: Trace) -> str:
    """Serialize a Trace into Med-PRM PRM-input format.

    Med-PRM expects each reasoning step followed by the separator ' ки' so
    the PRM logits can be read at those positions.  Final answer is the
    last line (conclusion step already contains it, but we add a canonical
    footer for ORM fallback).
    """
    rubric = load_rubric()
    sep = rubric["separator_token"]
    lines = []
    for s in trace.steps:
        # evidence inline in square brackets so the PRM sees them
        ev = f" [evidence: {', '.join(s.evidence_ids)}]" if s.evidence_ids else ""
        dir_tag = f" ({s.direction_tag})" if s.direction_tag != "n/a" else ""
        lines.append(f"Step {s.step_id}: {s.claim}{ev}{dir_tag}{sep}")
    ans = trace.final_answer
    lines.append(
        f"Final: family={ans.family}, subtype={ans.subtype}, "
        f"direction={ans.direction_tag}, polarity={ans.polarity}, "
        f"confidence={ans.confidence:.2f}, abstain={ans.abstain}{sep}"
    )
    if ans.summary:
        lines.append(f"Summary: {ans.summary}{sep}")
    return "\n".join(lines)


def trace_to_dict(trace: Trace) -> dict:
    return asdict(trace)


def trace_from_dict(d: dict) -> Trace:
    return Trace(
        pair_id=d["pair_id"],
        candidate_id=d["candidate_id"],
        steps=[Step(**s) for s in d["steps"]],
        final_answer=FinalAnswer(**d["final_answer"]),
        raw_text=d.get("raw_text", ""),
        provider=d.get("provider", ""),
        temperature=d.get("temperature", 0.0),
        generation_config=d.get("generation_config", {}),
    )


# ───────────────────────── helpers for B1b (prompt-side) ──────────────
EXAMPLE_JSON_SKELETON = """\
{
  "steps": [
    {
      "step_id": 1,
      "role": "pk_flag",
      "claim": "Drug A inhibits CYP3A4 (evidence: DB00497 flag cyp3a4_inh).",
      "evidence_ids": ["DB00497", "cyp3a4_inh"],
      "direction_tag": "a_to_b",
      "family_hint": "PK_Metabolism"
    },
    {
      "step_id": 2,
      "role": "mechanism_of_action",
      "claim": "Drug B is primarily metabolized by CYP3A4 (evidence: DB01234 moa; P08684).",
      "evidence_ids": ["DB01234", "P08684"],
      "direction_tag": "n/a",
      "family_hint": "PK_Metabolism"
    },
    {
      "step_id": 3,
      "role": "neighbor_pair",
      "claim": "Analogous pair DB00497|DB00284 (Ketoconazole, Aprepitant) has labeled PK_Metabolism/cyp3a4_inhibition.",
      "evidence_ids": ["DB00497|DB00284"],
      "direction_tag": "a_to_b",
      "family_hint": "PK_Metabolism"
    },
    {
      "step_id": 4,
      "role": "conclusion",
      "claim": "The metabolism of Drug B is decreased when combined with Drug A.",
      "evidence_ids": ["DB00497", "DB01234"],
      "direction_tag": "a_to_b",
      "polarity": "down"
    }
  ],
  "final_answer": {
    "family": "PK_Metabolism",
    "subtype": "metabolism",
    "direction_tag": "a_to_b",
    "polarity": "down",
    "confidence": 0.85,
    "abstain": false,
    "summary": "Drug A inhibits CYP3A4; Drug B is a CYP3A4 substrate, so co-administration decreases Drug B's metabolism (serum concentration rises)."
  }
}"""


# ───────────────────────── JSON extraction ─────────────────────────
_JSON_OBJ_RX = re.compile(r"\{.*\}", re.DOTALL)


def _normalize_trace_inplace(obj: dict) -> dict:
    """Canonicalize role names and polarity values so validators and
    downstream consumers see a tight vocabulary regardless of model drift.

    Applies ROLE_ALIASES and maps stringy polarity nulls ("none"/"null"/"")
    to Python None.  Operates in-place and returns the object for chaining.
    """
    if not isinstance(obj, dict):
        return obj
    for s in obj.get("steps") or []:
        if not isinstance(s, dict):
            continue
        role = s.get("role")
        if isinstance(role, str) and role in ROLE_ALIASES:
            s["role"] = ROLE_ALIASES[role]
        pol = s.get("polarity")
        if isinstance(pol, str) and pol.strip().lower() in _POLARITY_NULL_STRINGS:
            s["polarity"] = None
    fa = obj.get("final_answer")
    if isinstance(fa, dict):
        pol = fa.get("polarity")
        if isinstance(pol, str) and pol.strip().lower() in _POLARITY_NULL_STRINGS:
            fa["polarity"] = None
    return obj


def extract_json_block(text: str) -> dict | None:
    """Best-effort JSON parse from an LLM output.  Returns dict or None.

    Tries:
      1. pure text as JSON
      2. fenced ```json ... ``` block
      3. last {...} span in the text (greedy)

    All successful parses are normalized through `_normalize_trace_inplace`
    so role aliases and null-polarity strings become canonical before any
    validator or downstream consumer sees them.
    """
    text = text.strip()
    try:
        return _normalize_trace_inplace(json.loads(text))
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return _normalize_trace_inplace(json.loads(m.group(1)))
        except Exception:
            pass

    matches = _JSON_OBJ_RX.findall(text)
    for cand in reversed(matches):
        try:
            return _normalize_trace_inplace(json.loads(cand))
        except Exception:
            continue
    return None
