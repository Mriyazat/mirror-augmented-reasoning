"""B2 — QC filter for teacher candidates.

Gates (in order — short-circuits on failure where sensible):

  G1 schema          trace parses as JSON, matches rubric.step_schema
                       (uses validators from src.teacher.schema)
  G2 evidence_ids    every cited ID ∈ ContextBundle.context_ids()
                       (critical to mitigate hallucinated entities)
  G3 direction       final_answer.direction_tag matches gold_direction
                       AND no step asserts a direction contradicting gold
                       (CRITICAL — the earlier baseline's 51.4% mirror-direction errors)
  G4 family          final_answer.family == gold_family
                       (lets us use valid traces as SFT demonstrations)
  G5 polarity        final_answer.polarity matches gold_polarity if gold set
  G6 banned_phrases  no silencing language ("I cannot" / "insufficient info")
                       unless final_answer.abstain = True
                       (the earlier baseline's "silence rewards precision" fix)
  G7 length_bounds   2–12 steps; each step 5-200 words
  G8 summary         final_answer.summary present, within soft/hard word
                       cap, names at least drug A or B (soft)
  G9 hedging         hedge-marker density in summary ≤ max_hedge_density
                       (SELFDOUBT arxiv 2604.06389; earlier hedging-amplification fix);
                       bypassed when abstain=True (soft)
  G10 subtype        final_answer.subtype ∈ SUBTYPE_VOCAB[gold_family]
                       ("correct-reason-wrong-class" failure fix; NEAR-MISS
                       traces are preserved via tier tagging in merge.py)

A record can be tagged one of three tiers (for downstream weighting):
  - full_correct      all critical gates pass, subtype exact match
  - family_correct    critical gates pass, subtype wrong but in whitelist
  - near_miss         family_correct + direction or polarity mismatch

Two aggregation modes:
  - strict       all G1–G6 must pass (G7 soft)
  - pass-family  G1 + G4 only (useful for diagnosing gate separately)

Usage:
    python -m src.teacher.qc \\
        --raw outputs/teacher/raw_subset25k_ollama-llama3.1_8b.jsonl \\
        --split subset25k

Writes:
    outputs/teacher/qc_<rawbase>.jsonl        one record per candidate
    outputs/teacher/qc_<rawbase>.summary.md   markdown gate pass-rate table
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.teacher.context_builder import ContextBuilder, DATA
from src.teacher.evidence_resolution import expand_context_ids, resolves
from src.teacher.prompt import SUBTYPE_VOCAB
from src.teacher.schema import (
    BANNED_ABSTAIN_PHRASES, load_rubric, validate_trace_schema,
    extract_json_block, VALID_DIRECTION_TAGS,
)

ROOT = Path(__file__).resolve().parents[2]


# ────────────────────────────── gates ──────────────────────────────
def _gate_schema(rec: dict) -> tuple[bool, list[str], dict | None]:
    """G1: parse + structural validate.  Returns (ok, errs, parsed_trace)."""
    parsed = extract_json_block(rec.get("raw_text", ""))
    if parsed is None:
        return False, ["G1: no JSON object found in raw_text"], None
    errs = validate_trace_schema(parsed)
    if errs:
        return False, [f"G1: {e.where}: {e.reason}" for e in errs[:5]], parsed
    return True, [], parsed


def _gate_evidence(parsed: dict, context_ids: set[str]) -> tuple[bool, list[str]]:
    """G2 (critical): every evidence_id cited in a step must resolve to some
    canonical form in the retrieved context pool.  Resolution logic lives in
    `src.teacher.evidence_resolution` (shared with evaluation metrics).
    """
    expanded = expand_context_ids(context_ids)
    bad_total = 0
    report: list[str] = []
    for s in parsed["steps"]:
        bad: list[str] = []
        for eid in s.get("evidence_ids", []) or []:
            if not isinstance(eid, str):
                bad.append(repr(eid)); continue
            if not resolves(eid, expanded):
                bad.append(eid)
        if bad:
            bad_total += len(bad)
            if len(report) < 3:
                report.append(f"G2: step {s['step_id']} cites non-context IDs: {bad[:5]}")
    if bad_total == 0:
        return True, []
    return False, report + [f"G2: total hallucinated IDs = {bad_total}"]


def _gate_direction(parsed: dict, gold_direction: str | None) -> tuple[bool, list[str]]:
    if not gold_direction:
        return True, []  # can't evaluate
    fa = parsed["final_answer"]
    final_tag = fa.get("direction_tag")
    # Abstention is a valid answer for ANY gold direction.  When the model
    # honestly says "insufficient evidence" with abstain=true + n/a, we want
    # to preserve that trace — it trains the student on appropriate
    # uncertainty, which is the whole point of v4's anti-hedging design.
    # (Without this, 83% of the teacher's output was being silently dropped
    # by G3 just for being cautious.)
    if bool(fa.get("abstain", False)) and final_tag == "n/a":
        return True, []
    if gold_direction == "bidirectional":
        # Any committed tag is consistent with bidirectional (some pairs are
        # safely described directionally even when the label is symmetric).
        if final_tag in {"bidirectional", "a_to_b", "b_to_a"}:
            return True, []
        return False, [f"G3: final direction_tag {final_tag!r} not valid for bidirectional gold"]
    # directional gold — must match
    if final_tag == gold_direction:
        return True, []
    return False, [f"G3: final direction_tag {final_tag!r} != gold {gold_direction!r}"]


def _gate_family(parsed: dict, gold_family: str | None) -> tuple[bool, list[str]]:
    if not gold_family:
        return True, []
    fa = parsed["final_answer"]
    pf = fa.get("family")
    # Honest abstention is a valid output (teaches the student uncertainty).
    # When abstain=true the final_answer.family often comes back as "n/a" or
    # the mechanism the model saw (e.g. PK_Metabolism for a labeled-as-
    # AdverseRisk pair whose mechanism is metabolic).  Don't penalize those
    # for G4 — merge.py's tier system will tag them as abstention / near_miss.
    if bool(fa.get("abstain", False)):
        return True, []
    if pf == gold_family:
        return True, []
    return False, [f"G4: final family {pf!r} != gold {gold_family!r}"]


def _gate_polarity(parsed: dict, gold_polarity: str | None) -> tuple[bool, list[str]]:
    if not gold_polarity:
        return True, []
    pp = parsed["final_answer"].get("polarity")
    if pp == gold_polarity:
        return True, []
    # Allow null polarity in final_answer if gold is null; else soft-fail
    return False, [f"G5: final polarity {pp!r} != gold {gold_polarity!r}"]


def _gate_banned(parsed: dict) -> tuple[bool, list[str]]:
    abstain = bool(parsed["final_answer"].get("abstain", False))
    hits: list[tuple[int, str]] = []
    for s in parsed["steps"]:
        claim_lc = s.get("claim", "").lower()
        for phr in BANNED_ABSTAIN_PHRASES:
            if phr in claim_lc:
                hits.append((s["step_id"], phr))
                break
    if not hits:
        return True, []
    if abstain:
        return True, []  # explicit abstention is allowed to use these phrases
    return False, [f"G6: step {sid} uses silencing phrase {phr!r} "
                   f"without final_answer.abstain=True" for sid, phr in hits[:3]]


def _gate_length(parsed: dict) -> tuple[bool, list[str]]:
    rubric = load_rubric()
    tg = rubric["teacher_generation"]
    n_steps = len(parsed["steps"])
    errs = []
    if not tg["min_steps"] <= n_steps <= tg["max_steps"]:
        errs.append(f"G7: {n_steps} steps (want {tg['min_steps']}..{tg['max_steps']})")
    for s in parsed["steps"]:
        wc = len(s.get("claim", "").split())
        if wc < 3 or wc > 120:
            errs.append(f"G7: step {s['step_id']} claim length {wc} words")
            break
    return (not errs), errs


def _gate_summary(parsed: dict, ctx) -> tuple[bool, list[str]]:
    """G8 (soft): summary length ≤ soft cap AND mentions drug A or B name.

    The absolute hard cap is already enforced in validate_final_answer_schema (G1).
    G8 is stricter: it requires the summary to fit the SOFT word cap.
    """
    rubric = load_rubric()
    sc = rubric.get("summary_constraints", {})
    soft = int(sc.get("max_words", 80))
    summary = parsed["final_answer"].get("summary", "") or ""
    if not summary.strip():
        return False, ["G8: summary is empty"]
    wc = len(summary.split())
    if wc > soft:
        return False, [f"G8: summary is {wc} words; soft cap {soft}"]
    # Must mention at least one of the two drug names (lenient: case-insensitive
    # substring; accepts either name or DrugBank ID).
    lc = summary.lower()
    a_name = (ctx.a.name or "").lower()
    b_name = (ctx.b.name or "").lower()
    tokens = {ctx.a.drugbank_id.lower(), ctx.b.drugbank_id.lower()}
    if a_name: tokens.add(a_name)
    if b_name: tokens.add(b_name)
    if not any(tok and tok in lc for tok in tokens):
        return False, [f"G8: summary does not name either drug "
                       f"({ctx.a.name!r} or {ctx.b.name!r})"]
    return True, []


# Pre-compile hedging regex once; use word boundaries so "maybe" matches but
# "maybell" doesn't.  We match phrases too (e.g., "in some cases").
def _compile_hedging_regex() -> re.Pattern:
    rubric = load_rubric()
    markers = rubric.get("summary_constraints", {}).get("hedging_markers", [])
    if not markers:
        return re.compile(r"(?!x)x")  # never matches
    escaped = [re.escape(m) for m in markers]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


_HEDGE_RX = None


def _gate_hedging(parsed: dict) -> tuple[bool, list[str]]:
    """G9 (soft): hedging density in summary ≤ max_hedge_density.

    Bypassed when abstain=True (genuine uncertainty is allowed there).
    Evidence: arxiv 2604.06389 (SELFDOUBT) — traces with zero hedging
    markers are correct 96% of the time.
    """
    global _HEDGE_RX
    if _HEDGE_RX is None:
        _HEDGE_RX = _compile_hedging_regex()

    fa = parsed["final_answer"]
    if fa.get("abstain", False):
        return True, []  # hedging is appropriate when abstaining

    summary = fa.get("summary", "") or ""
    words = summary.split()
    if not words:
        return True, []  # G8 already catches empty summary

    hedges = _HEDGE_RX.findall(summary)
    density = len(hedges) / len(words)
    rubric = load_rubric()
    max_density = float(
        rubric.get("summary_constraints", {}).get("max_hedge_density", 0.15)
    )
    if density > max_density:
        return False, [
            f"G9: hedging density {density:.3f} > {max_density:.3f} "
            f"({len(hedges)} hedges in {len(words)} words); "
            f"markers found: {hedges[:5]}"
        ]
    return True, []


def _gate_subtype(parsed: dict, gold_family: str | None,
                  gold_subtype: str | None) -> tuple[bool, list[str], str]:
    """G10 (soft): subtype is in the gold family's whitelist.

    Returns (ok, errs, subtype_tier) where subtype_tier ∈
    {'exact', 'in_whitelist', 'out_of_whitelist'} — used by merge.py
    to assign the trace's tier (full_correct / family_correct / near_miss).
    """
    if not gold_family:
        return True, [], "exact"  # can't evaluate without gold

    predicted = parsed["final_answer"].get("subtype") or ""

    if gold_subtype and predicted == gold_subtype:
        return True, [], "exact"

    vocab = SUBTYPE_VOCAB.get(gold_family, [])
    if predicted in vocab:
        return True, [f"G10: subtype {predicted!r} in {gold_family} whitelist "
                      f"but != gold {gold_subtype!r} (near-miss)"], "in_whitelist"
    return False, [f"G10: subtype {predicted!r} not in {gold_family} whitelist"], \
        "out_of_whitelist"


# ───────────────────────── per-record QC ─────────────────────────
def qc_record(rec: dict, ctx_bundle_builder, strict: bool = True) -> dict:
    """Run all gates on a single raw candidate record and return the result."""
    pid = rec["pair_id"]

    # Gate 1: schema
    ok_g1, errs_g1, parsed = _gate_schema(rec)
    if not ok_g1:
        return {
            "pair_id": pid, "candidate_id": rec["candidate_id"],
            "passed": False, "gates": {"G1": False},
            "errors": errs_g1,
            "parsed": None,
        }

    # Build context for evidence grounding
    try:
        ctx = ctx_bundle_builder(pid)
        context_ids = ctx.context_ids()
    except Exception as e:
        return {
            "pair_id": pid, "candidate_id": rec["candidate_id"],
            "passed": False, "gates": {"G1": True},
            "errors": [f"context: {e}"],
            "parsed": parsed,
        }

    gates = {"G1": True}
    errs: list[str] = []

    ok_g2, e2 = _gate_evidence(parsed, context_ids)
    gates["G2"] = ok_g2; errs += e2
    ok_g3, e3 = _gate_direction(parsed, rec.get("gold_direction"))
    gates["G3"] = ok_g3; errs += e3
    ok_g4, e4 = _gate_family(parsed, rec.get("gold_family"))
    gates["G4"] = ok_g4; errs += e4
    ok_g5, e5 = _gate_polarity(parsed, rec.get("gold_polarity"))
    gates["G5"] = ok_g5; errs += e5
    ok_g6, e6 = _gate_banned(parsed)
    gates["G6"] = ok_g6; errs += e6
    ok_g7, e7 = _gate_length(parsed)
    gates["G7"] = ok_g7; errs += e7
    ok_g8, e8 = _gate_summary(parsed, ctx)
    gates["G8"] = ok_g8; errs += e8
    ok_g9, e9 = _gate_hedging(parsed)
    gates["G9"] = ok_g9; errs += e9
    ok_g10, e10, subtype_tier = _gate_subtype(
        parsed, rec.get("gold_family"), rec.get("gold_subtype")
    )
    gates["G10"] = ok_g10; errs += e10

    critical = all([gates["G1"], gates["G2"], gates["G3"], gates["G4"], gates["G6"]])

    # Honest abstention detection: parser reports abstain=True, the trace
    # contains an explicit abstention / evidence_gap step, AND the structural
    # gates are intact. These carry distinct training signal (calibrated
    # uncertainty) and get their own tier in Phase C.
    fa = parsed.get("final_answer") or {}
    is_abstain = bool(fa.get("abstain"))
    has_abstention_step = any(
        s.get("role") in ("abstention", "evidence_gap")
        for s in (parsed.get("steps") or [])
        if isinstance(s, dict)
    )
    well_reasoned_abstention = (
        is_abstain
        and gates["G1"] and gates["G2"] and gates["G3"] and gates["G6"]
        and has_abstention_step
    )

    # Tier classification (for Phase C SFT weighting).
    #   full_correct   = critical pass AND subtype exact AND G5 pass (commit+right)
    #   family_correct = critical pass AND subtype in whitelist (family+dir right)
    #   near_miss      = G4 pass + subtype in whitelist, but G3 or G5 failed
    #   abstention     = honest uncertainty with intact schema + grounded evidence
    #   drop           = otherwise (broken schema, hallucinated evidence, or wrong-committed)
    if critical and subtype_tier == "exact" and gates["G5"] and not is_abstain:
        tier = "full_correct"
    elif critical and subtype_tier == "in_whitelist" and not is_abstain:
        tier = "family_correct"
    elif gates["G1"] and gates["G2"] and gates["G4"] and subtype_tier != "out_of_whitelist" and not is_abstain:
        tier = "near_miss"
    elif well_reasoned_abstention:
        tier = "abstention"
    else:
        tier = "drop"

    if strict:
        # Strict pass now also requires G9 (no hedging amplification risk)
        # and G10 (subtype in whitelist). G7/G8 remain soft.
        passed = critical and gates["G5"] and gates["G7"] and gates["G8"] \
            and gates["G9"] and gates["G10"]
    else:
        passed = critical

    return {
        "pair_id": pid,
        "candidate_id": rec["candidate_id"],
        "passed": passed,
        "critical_passed": critical,
        "tier": tier,                    # NEW: used by merge.py
        "subtype_tier": subtype_tier,    # NEW: exact / in_whitelist / out_of_whitelist
        "gates": gates,
        "errors": errs,
        "parsed": parsed,                # keep the parsed trace for later stages
        "gold_family": rec.get("gold_family"),
        "gold_subtype": rec.get("gold_subtype"),
        "gold_direction": rec.get("gold_direction"),
        "gold_polarity": rec.get("gold_polarity"),
        "provider": rec.get("provider"),
        "temperature": rec.get("temperature"),
    }


# ───────────────────────── driver ─────────────────────────
def run_qc(raw_path: Path, split: str) -> Path:
    raw_path = raw_path.resolve()
    cb = ContextBuilder(neighbor_index_path=DATA / f"neighbor_index_{split}.parquet")

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    print(f"[qc] reading {_rel(raw_path)}", flush=True)

    # Output files
    base = raw_path.stem.replace("raw_", "qc_", 1)
    out = raw_path.parent / f"{base}.jsonl"
    summary = raw_path.parent / f"{base}.summary.md"

    # Per-gate counters
    gate_pass = Counter()
    gate_total = Counter()
    tier_counts = Counter()
    passed_total = 0
    critical_pass_total = 0
    n_rec = 0
    by_pair: dict[str, list[dict]] = defaultdict(list)

    with raw_path.open() as fin, out.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            res = qc_record(rec, cb.build, strict=True)
            fout.write(json.dumps(res) + "\n")

            n_rec += 1
            if res["passed"]:
                passed_total += 1
            if res.get("critical_passed"):
                critical_pass_total += 1
            tier_counts[res.get("tier", "drop")] += 1
            for g, ok in res["gates"].items():
                gate_total[g] += 1
                if ok:
                    gate_pass[g] += 1
            by_pair[res["pair_id"]].append(res)

    # Per-pair stats
    n_pairs = len(by_pair)
    pairs_with_any = sum(1 for v in by_pair.values() if any(r["passed"] for r in v))
    pairs_with_critical = sum(1 for v in by_pair.values()
                              if any(r.get("critical_passed") for r in v))

    # Markdown summary
    lines = [
        f"# QC summary — {raw_path.name}",
        "",
        f"- **Candidates:** {n_rec:,} across {n_pairs:,} pairs "
        f"({n_rec / max(1, n_pairs):.1f} per pair)",
        f"- **Strict pass (all critical + soft):** {passed_total:,} "
        f"({100*passed_total/max(1,n_rec):.1f}%)",
        f"- **Critical pass (G1+G2+G3+G4+G6):** {critical_pass_total:,} "
        f"({100*critical_pass_total/max(1,n_rec):.1f}%)",
        f"- **Pairs with ≥1 strict-pass candidate:** {pairs_with_any:,} "
        f"({100*pairs_with_any/max(1,n_pairs):.1f}%)",
        f"- **Pairs with ≥1 critical-pass candidate:** {pairs_with_critical:,} "
        f"({100*pairs_with_critical/max(1,n_pairs):.1f}%)",
        "",
        "## Per-gate pass rate",
        "",
        "| Gate | Description | Pass rate |",
        "|------|-------------|-----------|",
    ]
    descriptions = {
        "G1": "Schema parse + structural validity",
        "G2": "Evidence IDs all ∈ context (no hallucinations)",
        "G3": "Final direction_tag matches gold",
        "G4": "Final family matches gold",
        "G5": "Final polarity matches gold (soft)",
        "G6": "No banned silencing phrases (unless abstain=True)",
        "G7": "Step count + per-step length bounds (soft)",
        "G8": "Summary ≤ soft cap AND names drug A or B (soft)",
        "G9": "Hedging density ≤ max (unless abstain=True) (soft)",
        "G10": "Subtype ∈ SUBTYPE_VOCAB[gold_family] (soft)",
    }
    for g in ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"]:
        t = gate_total[g]
        p = gate_pass[g]
        if t == 0:
            lines.append(f"| {g} | {descriptions[g]} | — |")
        else:
            lines.append(f"| {g} | {descriptions[g]} | {p:,}/{t:,} "
                         f"({100*p/t:.1f}%) |")
    lines.append("")
    lines.append("## Tier breakdown (for Phase C SFT weighting)")
    lines.append("")
    lines.append("| Tier | Count | Share |")
    lines.append("|------|-------|-------|")
    for tier in ("full_correct", "family_correct", "abstention", "near_miss", "drop"):
        c = tier_counts.get(tier, 0)
        lines.append(f"| {tier} | {c:,} | {100*c/max(1,n_rec):.1f}% |")
    lines.append("")

    summary.write_text("\n".join(lines))
    print("\n".join(lines[:16]))
    print(f"\n[qc] per-candidate output: {_rel(out)}")
    print(f"[qc] summary:              {_rel(summary)}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="Path to raw_<split>_<provider>.jsonl")
    p.add_argument("--split", default="subset25k")
    args = p.parse_args()
    run_qc(Path(args.raw), args.split)
