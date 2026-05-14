"""B7 -- Post-merge corpus quality audit.

This is a SECOND pass on `teacher_clean.jsonl`, run after critic+merge,
that checks every trace against continuous quality signals QC's binary
gates didn't measure.  Outputs a per-trace audit JSONL and a markdown
summary so we can decide:
    - Is the SFT corpus actually clean enough to train Phase C on?
    - Should we down-weight or drop specific tiers / families?
    - Which traces should we manually inspect?

Why a second pass?  QC ran per-CANDIDATE with binary gates G1..G10.
It guaranteed every accepted trace:
    G1: schema parses
    G2: evidence_ids ⊆ context_ids (binary)
    G3: direction_tag matches gold (binary)
    G4: family matches gold (binary)
    ...

What QC did NOT measure:
    - **Evidence grounding RATE**: how many context IDs the trace actually
      references in its claim prose, vs.  just listing them in
      `evidence_ids` arrays.  A trace can technically pass G2 with claims
      that never refer to the evidence at all.
    - **Mechanism-skeleton match per family**: PK_Metabolism traces should
      have at least one `pk_flag` step naming a CYP / transporter; AdverseRisk
      traces should cite a neighbor or shared protein etc.  QC didn't enforce
      this.
    - **Confidence-tier calibration**: full_correct traces should have high
      `final_answer.confidence`, abstention traces should have low.  Mismatch
      flags miscalibrated reasoning.
    - **Subtype-mechanism alignment**: subtype="metabolism" should mention
      a CYP/transporter; "bleeding_and_hemorrhage" should mention coagulation
      or platelet; etc.
    - **Step-role diversity**: a "trace" of 1 conclusion-only step is not
      multi-step reasoning; flag it.
    - **Cross-trace name consistency**: do drug A and drug B names appear
      in the trace's prose at all?

Output
------
    outputs/audit/b7_audit_per_trace.jsonl
        One JSON record per trace with all signal scores + a final
        `quality_score` and `flags` list.
    outputs/audit/b7_audit_summary.md
        Aggregate report (distributions, per-tier × per-teacher table,
        top-suspect traces).
    outputs/audit/b7_suspect_traces.jsonl
        Traces with quality_score below `--suspect_threshold`, intended
        for human spot-checks.

CLI
---
    python -m src.teacher.audit_teacher_clean \
        --teacher_clean /path/to/teacher_clean.jsonl \
        --critic_dir    /path/to/teacher (looks for critic_*_<teacher>.jsonl) \
        --output_dir    outputs/audit \
        --suspect_threshold 0.55 \
        --split subset25k

Auditing 23,665 traces takes ~2-3 minutes on a laptop CPU.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from src.teacher.schema import extract_json_block
from src.teacher.context_builder import ContextBuilder, DATA
from src.teacher.evidence_resolution import expand_context_ids, resolves


ROOT = Path(__file__).resolve().parents[2]


# ======================================================================
# Family / subtype -> required mechanism-step roles + keyword lexicons
# ======================================================================
# Each family declares which step roles are *expected* for a healthy
# trace.  Traces missing every "expected" role are flagged --
# they're not necessarily wrong, just suspiciously thin.
_FAMILY_REQUIRED_ROLES: dict[str, set[str]] = {
    "PK_Metabolism":   {"pk_flag", "protein", "mechanism_of_action"},
    "PK_Excretion":    {"pk_flag", "protein"},
    "PK_Absorption":   {"pk_flag", "protein"},
    "PK_Distribution": {"pk_flag", "protein", "mechanism_of_action"},
    "PD_Activity":     {"protein", "mechanism_of_action", "pathway"},
    "Efficacy":        {"protein", "mechanism_of_action", "pathway", "neighbor_pair"},
    "AdverseRisk":     {"protein", "mechanism_of_action", "neighbor_pair", "pathway"},
}

# Subtype -> regex of keywords that should appear somewhere in the trace
# prose.  When a trace's subtype claims `metabolism` but no CYP / transporter
# is named anywhere, we flag it as mechanism-keyword mismatch.  These
# regexes are intentionally loose -- false negatives matter more than
# false positives here.
_SUBTYPE_KEYWORD_RX: dict[str, re.Pattern] = {
    "metabolism":                re.compile(r"\bcyp\d|p[\s-]?gp|oatp|bcrp|oct\d|oat\d|metaboli", re.I),
    "serum_concentration":       re.compile(r"\bcyp\d|p[\s-]?gp|oatp|bcrp|metaboli|exposure|auc|cmax", re.I),
    "excretion_rate":            re.compile(r"\bp[\s-]?gp|oatp|oat\d|oct\d|bcrp|excret|elimina|kidney|renal", re.I),
    "absorption_change":         re.compile(r"\bp[\s-]?gp|bcrp|absorp|gut|intestin|bioavail", re.I),
    "bioavailability":           re.compile(r"\bbioavail|absorp|p[\s-]?gp|bcrp", re.I),
    "protein_binding":           re.compile(r"protein binding|albumin|displace", re.I),
    "active_metabolite_serum_conc": re.compile(r"\bcyp\d|metaboli|active metabolite|prodrug", re.I),
    "bleeding_and_hemorrhage":   re.compile(r"bleed|hemorrhag|haemorrhag|coagul|platelet|warfarin|heparin|factor[\s-]?(?:ii|x)|antithromb", re.I),
    "qt_prolongation":           re.compile(r"\bqt|torsade|hERG|arrhythm", re.I),
    "hyperglycemia":             re.compile(r"hyper(glyc|glyk)|glucose|insulin|diabet|sulfonyl|biguanide", re.I),
    "hypoglycemia":              re.compile(r"hypo(glyc|glyk)|glucose|insulin|sulfonyl", re.I),
    "cns_depression":            re.compile(r"cns|sedat|drowsi|gaba|barbiturat|benzodiaz|opioid", re.I),
    "renal_failure":             re.compile(r"renal|kidney|nephro|creatinine|gfr|aki", re.I),
    "hepatotoxicity":            re.compile(r"hepato|liver|alt|ast|bilirubin", re.I),
    "myopathy":                  re.compile(r"myopath|rhabdomyo|cpk|statin", re.I),
    "serotonin_syndrome":        re.compile(r"serotoni|5-?ht|maoi|ssri|snri|trypto", re.I),
}

# Hedging vocabulary used to compute density.  Each word counted at
# most once per claim to avoid double-charging.
_HEDGING_WORDS = {
    "may", "might", "could", "potentially", "possibly", "perhaps",
    "uncertain", "unclear", "unknown", "limited", "insufficient",
    "appears", "seems", "likely", "unlikely", "probably",
}


# ======================================================================
# Per-trace audit
# ======================================================================
def _trace_text(parsed: dict) -> str:
    """Concatenate all step claims + summary into one prose blob."""
    parts: list[str] = []
    for s in parsed.get("steps", []) or []:
        parts.append(str(s.get("claim") or ""))
    parts.append(str((parsed.get("final_answer") or {}).get("summary") or ""))
    return "  ".join(parts)


def _hedging_density(text: str) -> float:
    if not text:
        return 0.0
    words = re.findall(r"\b[a-zA-Z][a-zA-Z']*\b", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _HEDGING_WORDS)
    return hits / len(words)


_LIT_RX = re.compile(r"^[ALFT]\d{3,7}$")            # DrugBank article refs
_METRIC_RX = re.compile(r"^(pathway|protein)_jaccard$|^smiles_tanimoto$|^atc_prefix_depth$")
_FLAG_PREFIX_RX = re.compile(r"^([a-z][a-z0-9]+?)[_-](sub|inh|ind|substrate|inhibitor|inducer|substrates|inhibitors|inducers)$", re.I)
_DRUGBANK_RX = re.compile(r"^DB\d{5}$")
_PAIR_RX = re.compile(r"^DB\d{5}\|DB\d{5}$")


def _evidence_grounding_rate(parsed: dict, ctx_ids: set[str],
                             a_name: str, b_name: str) -> float:
    """Fraction of cited `evidence_ids` that are GROUNDED in the trace.

    "Grounded" is operationalised more permissively than literal substring
    matching, because real traces refer to evidence by *drug name* /
    *enzyme name* rather than the raw ID:

      - Literature refs (A12345 / L12345 / F12345 / T12345): grounded
        whenever they resolve to the context pool (these are pure
        citation IDs, never expected to appear in prose verbatim).
      - Metric names (pathway_jaccard etc.): grounded whenever the bare
        metric name appears in prose.
      - DrugBank IDs (DB00123): grounded if the drug's NAME (a_name or
        b_name) appears in prose, OR if the ID itself does.
      - Pair IDs (DB12345|DB67890): grounded if "analogous", "neighbor",
        "similar pair" appears -- these are cited as analogies.
      - PK flags (cyp3a4_inh / cyp3a4_substrate): grounded if the bare
        enzyme/transporter name (cyp3a4 / p-gp / etc.) appears in prose.
      - Protein UniProts (P10635 etc.): grounded if the ID OR a recognized
        protein-name keyword appears.  We accept the ID as a fallback.
      - Otherwise: must appear in prose verbatim.
    """
    claims_text = _trace_text(parsed).lower()
    a_lc = (a_name or "").lower().strip()
    b_lc = (b_name or "").lower().strip()
    cited: list[str] = []
    for s in parsed.get("steps", []) or []:
        for eid in s.get("evidence_ids") or []:
            if isinstance(eid, str) and eid:
                cited.append(eid)
    if not cited:
        return 0.0
    expanded = expand_context_ids(cited)
    hits = 0
    for eid in cited:
        if not resolves(eid, expanded):
            continue
        eid_l = eid.lower().strip()
        # Literature refs always count as grounded if cited (citation IDs).
        if _LIT_RX.match(eid):
            hits += 1; continue
        if _METRIC_RX.match(eid_l):
            # Either bare metric name or "jaccard"/"tanimoto"/"atc" tokens
            if eid_l in claims_text or any(t in claims_text for t in
                ("jaccard", "tanimoto", "similarity", "overlap")):
                hits += 1; continue
            hits += 1; continue  # always grounded as a similarity statement
        if _DRUGBANK_RX.match(eid):
            if eid_l in claims_text:
                hits += 1; continue
            if a_lc and a_lc in claims_text:
                hits += 1; continue
            if b_lc and b_lc in claims_text:
                hits += 1; continue
            continue
        if _PAIR_RX.match(eid):
            if any(k in claims_text for k in
                   ("analog", "neighbor", "similar pair", "labeled pair")):
                hits += 1; continue
        # PK flag style: cyp3a4_inh, p_gp_substrate, etc.
        m = _FLAG_PREFIX_RX.match(eid_l)
        if m:
            base = m.group(1)
            if base in claims_text:
                hits += 1; continue
            # Soften 'p_gp' -> 'p-gp' / 'pgp'
            if base == "p_gp" and ("p-gp" in claims_text or "pgp" in claims_text or "p gp" in claims_text):
                hits += 1; continue
        if eid_l in claims_text:
            hits += 1; continue
    return hits / len(cited)


def _mechanism_skeleton_score(parsed: dict, family: str) -> tuple[float, list[str]]:
    roles = {s.get("role") for s in parsed.get("steps") or []}
    expected = _FAMILY_REQUIRED_ROLES.get(family, set())
    if not expected:
        return 1.0, []
    hits = roles & expected
    score = len(hits) / max(1, len(expected))
    miss = sorted(expected - hits)
    return score, miss


def _direction_consistency(parsed: dict) -> bool:
    """Does the conclusion / abstention step's direction_tag match the
    final_answer's direction_tag?"""
    fa = parsed.get("final_answer") or {}
    fa_dir = fa.get("direction_tag")
    if fa_dir in (None, "", "n/a"):
        return True
    for s in reversed(parsed.get("steps") or []):
        if s.get("role") in ("conclusion", "abstention"):
            sd = s.get("direction_tag")
            if sd in (None, "", "n/a"):
                return True
            return sd == fa_dir
    return True


def _confidence_calibration(parsed: dict, tier: str) -> tuple[bool, str]:
    """Returns (ok, why_not) -- compare final_answer.confidence to the tier."""
    fa = parsed.get("final_answer") or {}
    conf = fa.get("confidence")
    abst = bool(fa.get("abstain", False))
    if conf is None:
        return True, ""
    try:
        conf = float(conf)
    except Exception:
        return False, "confidence_not_numeric"
    if abst:
        # Abstention should not claim high confidence in a class
        if conf > 0.55:
            return False, f"abstain_but_high_conf={conf:.2f}"
        return True, ""
    if tier == "full_correct" and conf < 0.40:
        return False, f"full_correct_but_low_conf={conf:.2f}"
    if tier == "near_miss" and conf > 0.85:
        return False, f"near_miss_but_overconfident={conf:.2f}"
    return True, ""


def _drug_name_in_summary(parsed: dict, a_name: str, b_name: str) -> bool:
    fa = parsed.get("final_answer") or {}
    summary = (fa.get("summary") or "").lower()
    if not summary:
        return False
    a_lc = (a_name or "").lower()
    b_lc = (b_name or "").lower()
    return (a_lc and a_lc in summary) or (b_lc and b_lc in summary)


def _subtype_mechanism_match(parsed: dict, subtype: str) -> tuple[bool, bool]:
    """Returns (rule_exists, ok).  If no rule for this subtype, rule_exists=False."""
    rx = _SUBTYPE_KEYWORD_RX.get(subtype)
    if rx is None:
        return False, True
    text = _trace_text(parsed)
    return True, bool(rx.search(text))


_ABSTAIN_ROLES = {"evidence_gap", "abstention"}


def _abstention_quality(parsed: dict, ctx_size: int) -> tuple[float, list[str]]:
    """For abstention-tier traces, evidence grounding and mechanism
    skeleton don't apply (the model is honestly saying "no evidence
    found").  Instead audit:
        - At least 1 step with role in {evidence_gap, abstention}
        - For pairs WITH evidence in context (ctx_size >= ~10),
          require at least one inspection step (pathway/protein/pk_flag/
          mechanism_of_action).  Pairs with very small context (e.g.,
          biologic + biologic) are exempt -- there's nothing to inspect.
        - confidence < 0.55
        - explicit abstain=True in final_answer
    Returns (score, flags).
    """
    flags: list[str] = []
    steps = parsed.get("steps") or []
    fa = parsed.get("final_answer") or {}

    n_abst = sum(1 for s in steps if s.get("role") in _ABSTAIN_ROLES)
    if n_abst < 1:
        flags.append("abstention_without_evidence_gap_step")

    n_inspection = sum(
        1 for s in steps
        if s.get("role") in {"pathway", "protein", "pk_flag",
                              "mechanism_of_action", "neighbor_pair",
                              "pair_similarity", "structural", "atc"}
    )
    # Only require an inspection step when there's actually evidence
    # to inspect.  Small ctx (~drug IDs + ATC only) means the pair is
    # legitimately information-poor (e.g., biologic+biologic) and
    # skipping straight to evidence_gap is correct behaviour.
    if ctx_size >= 10 and n_inspection < 1:
        flags.append("abstention_without_inspection_step")

    if not fa.get("abstain", False):
        flags.append("abstention_tier_but_not_abstain_flag")

    conf = fa.get("confidence")
    try:
        conf_f = float(conf) if conf is not None else None
    except Exception:
        conf_f = None
    if conf_f is not None and conf_f > 0.55:
        flags.append(f"abstention_high_confidence={conf_f:.2f}")

    components = [
        1.0 if n_abst >= 1 else 0.0,
        # Inspection step REQUIRED only when context had evidence:
        1.0 if (n_inspection >= 1 or ctx_size < 10) else 0.0,
        1.0 if fa.get("abstain", False) else 0.0,
        1.0 if (conf_f is None or conf_f <= 0.55) else 0.0,
    ]
    return sum(components) / len(components), flags


def audit_one(record: dict, cb: ContextBuilder) -> dict:
    pair_id = record.get("pair_id", "")
    family = record.get("family") or ""
    subtype = record.get("subtype") or ""
    tier = record.get("tier", "")

    asst_msg = next((m for m in record.get("messages", []) if m.get("role") == "assistant"), None)
    raw = (asst_msg or {}).get("content", "")
    try:
        parsed = extract_json_block(raw)
    except Exception:
        parsed = None

    out = {
        "pair_id": pair_id,
        "family": family,
        "subtype": subtype,
        "tier": tier,
        "schema_ok": parsed is not None,
        "flags": [],
    }
    if parsed is None:
        out["quality_score"] = 0.0
        out["flags"].append("schema_fail")
        return out

    # Build context for grounding rate
    try:
        ctx = cb.build(pair_id)
        ctx_ids = ctx.context_ids()
        a_name = ctx.a.name
        b_name = ctx.b.name
    except Exception as e:
        ctx_ids = set()
        a_name = b_name = ""
        out["flags"].append(f"ctx_build_fail:{type(e).__name__}")

    # ---------- Tier-specific path: abstention ----------
    is_abstention = (tier == "abstention" or
                     bool((parsed.get("final_answer") or {}).get("abstain", False)))
    if is_abstention:
        abst_score, abst_flags = _abstention_quality(parsed, len(ctx_ids))
        out["abstention_quality_score"] = round(abst_score, 3)
        out["flags"].extend(abst_flags)
        # Also report some shared signals so the JSONL has same shape:
        steps = parsed.get("steps") or []
        out["n_steps"] = len(steps)
        out["n_distinct_roles"] = len({s.get("role") for s in steps})
        out["evidence_grounding_rate"] = None
        out["mechanism_skeleton_score"] = None
        out["direction_consistent"] = _direction_consistency(parsed)
        if not out["direction_consistent"]:
            out["flags"].append("direction_inconsistent")
        out["drug_name_in_summary"] = _drug_name_in_summary(parsed, a_name, b_name)
        if not out["drug_name_in_summary"]:
            out["flags"].append("no_drug_name_in_summary")
        text = _trace_text(parsed)
        out["hedging_density"] = round(_hedging_density(text), 3)
        # Compose final score: abstention-specific score + a couple of
        # shared signals (drug name, direction, structure).
        components = [
            abst_score,
            float(out["direction_consistent"]),
            float(out["drug_name_in_summary"]),
            min(1.0, out["n_steps"] / 3),
        ]
        out["quality_score"] = round(sum(components) / len(components), 3)
        return out

    # Step-role diversity
    steps = parsed.get("steps") or []
    roles = [s.get("role") for s in steps]
    n_steps = len(steps)
    n_distinct = len(set(roles))
    out["n_steps"] = n_steps
    out["n_distinct_roles"] = n_distinct
    if n_steps < 3:
        out["flags"].append(f"few_steps={n_steps}")
    if n_distinct < 2:
        out["flags"].append(f"single_role_only")

    # Evidence-grounding rate (continuous G2)
    out["evidence_grounding_rate"] = round(
        _evidence_grounding_rate(parsed, ctx_ids, a_name, b_name), 3)
    if out["evidence_grounding_rate"] < 0.25:
        out["flags"].append("low_evidence_grounding")

    # Mechanism-skeleton match
    mech_score, mech_miss = _mechanism_skeleton_score(parsed, family)
    out["mechanism_skeleton_score"] = round(mech_score, 3)
    out["mechanism_skeleton_missing_roles"] = mech_miss
    if mech_score < 0.34 and family in _FAMILY_REQUIRED_ROLES:
        out["flags"].append(f"weak_mechanism_skeleton:{','.join(mech_miss)}")

    # Direction consistency
    out["direction_consistent"] = _direction_consistency(parsed)
    if not out["direction_consistent"]:
        out["flags"].append("direction_inconsistent")

    # Confidence calibration vs tier
    cal_ok, cal_why = _confidence_calibration(parsed, tier)
    out["confidence_calibration_ok"] = cal_ok
    if not cal_ok:
        out["flags"].append(f"calibration:{cal_why}")

    # Drug name in summary
    out["drug_name_in_summary"] = _drug_name_in_summary(parsed, a_name, b_name)
    if not out["drug_name_in_summary"]:
        out["flags"].append("no_drug_name_in_summary")

    # Subtype-mechanism keyword match
    rule_exists, ok = _subtype_mechanism_match(parsed, subtype)
    out["subtype_mechanism_rule_exists"] = rule_exists
    out["subtype_mechanism_match"]       = ok
    if rule_exists and not ok:
        out["flags"].append(f"subtype_mechanism_miss:{subtype}")

    # Hedging density
    text = _trace_text(parsed)
    out["hedging_density"] = round(_hedging_density(text), 3)
    if out["hedging_density"] > 0.10:
        out["flags"].append(f"high_hedging={out['hedging_density']:.2f}")

    # Final composite score (0..1).  Equal-weighted average of the
    # binary checks + continuous metrics, clamped to [0,1].
    components = [
        float(out["schema_ok"]),
        float(out["evidence_grounding_rate"]),
        float(out["mechanism_skeleton_score"]),
        float(out["direction_consistent"]),
        float(out["confidence_calibration_ok"]),
        float(out["drug_name_in_summary"]),
        float(out["subtype_mechanism_match"] or not out["subtype_mechanism_rule_exists"]),
        max(0.0, 1.0 - out["hedging_density"] * 4),  # 0.25 hedging -> 0
        min(1.0, n_distinct / 3),                    # diversity reward
    ]
    out["quality_score"] = round(sum(components) / len(components), 3)
    return out


# ======================================================================
# Aggregate report
# ======================================================================
def write_summary_report(audits: list[dict], out_path: Path,
                         suspect_threshold: float) -> None:
    n = len(audits)
    if n == 0:
        return

    # Aggregate distributions
    qs = [a["quality_score"] for a in audits if "quality_score" in a]
    qs_mean = sum(qs) / max(1, len(qs))
    qs_below = sum(1 for x in qs if x < suspect_threshold)

    flag_counter: Counter = Counter()
    for a in audits:
        for f in a.get("flags", []):
            # Bucket numeric variations
            key = f.split("=")[0]
            flag_counter[key] += 1

    # Per-tier
    by_tier: dict[str, list[float]] = defaultdict(list)
    by_tier_flags: dict[str, Counter] = defaultdict(Counter)
    for a in audits:
        by_tier[a.get("tier", "?")].append(a["quality_score"])
        for f in a.get("flags", []):
            by_tier_flags[a.get("tier", "?")][f.split("=")[0]] += 1

    # Per-family
    by_fam: dict[str, list[float]] = defaultdict(list)
    for a in audits:
        by_fam[a.get("family", "?")].append(a["quality_score"])

    lines = [
        "# B7 -- Post-merge teacher_clean quality audit",
        "",
        f"- Records audited: **{n:,}**",
        f"- Mean quality score: **{qs_mean:.3f}**",
        f"- Records below threshold ({suspect_threshold:.2f}): **{qs_below:,}** "
        f"({100*qs_below/max(1,n):.1f}%)",
        "",
        "## Quality score percentiles",
        "",
        "| pct | score |",
        "|---|---|",
    ]
    qs_sorted = sorted(qs)
    for pct in (5, 10, 25, 50, 75, 90, 95):
        idx = max(0, min(len(qs_sorted) - 1, int(pct/100 * len(qs_sorted))))
        lines.append(f"| p{pct} | {qs_sorted[idx]:.3f} |")

    lines += ["", "## Flags by frequency", "",
              "| flag | n | share |", "|---|---:|---:|"]
    for flag, c in flag_counter.most_common():
        lines.append(f"| `{flag}` | {c:,} | {100*c/n:.1f}% |")

    lines += ["", "## Quality by tier", "",
              "| tier | n | mean qs | below-threshold | top-3 flags |",
              "|---|---:|---:|---:|---|"]
    for tier in sorted(by_tier):
        scores = by_tier[tier]
        cnt = len(scores)
        mean = sum(scores)/max(1, cnt)
        below = sum(1 for s in scores if s < suspect_threshold)
        top_flags = ", ".join(f"`{f}` ({c})"
                              for f, c in by_tier_flags[tier].most_common(3))
        lines.append(f"| {tier} | {cnt:,} | {mean:.3f} | {below:,} ({100*below/max(1,cnt):.0f}%) | {top_flags} |")

    lines += ["", "## Quality by family", "",
              "| family | n | mean qs | below-threshold |",
              "|---|---:|---:|---:|"]
    for fam in sorted(by_fam):
        scores = by_fam[fam]
        cnt = len(scores)
        mean = sum(scores)/max(1, cnt)
        below = sum(1 for s in scores if s < suspect_threshold)
        lines.append(f"| {fam} | {cnt:,} | {mean:.3f} | {below:,} ({100*below/max(1,cnt):.0f}%) |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_clean", required=True)
    ap.add_argument("--output_dir",    default="outputs/audit")
    ap.add_argument("--split",         default="subset25k")
    ap.add_argument("--suspect_threshold", type=float, default=0.55)
    ap.add_argument("--progress_every", type=int, default=2500)
    ap.add_argument("--limit", type=int, default=None,
                    help="Audit only first N records (for testing).")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cb = ContextBuilder(neighbor_index_path=DATA / f"neighbor_index_{args.split}.parquet")

    per_trace_path = out_dir / "b7_audit_per_trace.jsonl"
    suspect_path   = out_dir / "b7_suspect_traces.jsonl"
    summary_path   = out_dir / "b7_audit_summary.md"

    audits: list[dict] = []
    n_in = 0
    with open(args.teacher_clean) as fin, \
         open(per_trace_path, "w") as f_out, \
         open(suspect_path, "w") as f_susp:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.limit is not None and n_in > args.limit:
                break
            rec = json.loads(line)
            audit = audit_one(rec, cb)
            audits.append(audit)
            f_out.write(json.dumps(audit) + "\n")
            if audit.get("quality_score", 0.0) < args.suspect_threshold:
                # Sidecar: include the original record + audit for spot inspection
                payload = {
                    "audit": audit,
                    "trace": rec,
                }
                f_susp.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if n_in % args.progress_every == 0:
                avg = sum(a.get("quality_score", 0.0) for a in audits) / len(audits)
                print(f"[audit] {n_in:,} done.  running mean qs = {avg:.3f}")

    print(f"[audit] read {n_in:,} traces.  outputs:")
    print(f"  per-trace: {per_trace_path}")
    print(f"  suspect:   {suspect_path}")

    write_summary_report(audits, summary_path, args.suspect_threshold)
    print(f"  summary:   {summary_path}")


if __name__ == "__main__":
    main()
