"""B6 -- Build mirror preference pairs for Phase C3 (Mirror-DPO/IPO).

Input
-----
    outputs/teacher/teacher_clean.jsonl       # winners only, tier-annotated
    outputs/teacher/prm_scores.jsonl          # optional; per-step PRM scores

Output
------
    outputs/preferences/mirror_pairs.jsonl    # one preference pair per record
    outputs/audit/b6_preference_report.md     # histogram of pair types

Preference-pair format (consumed by `src/training/dpo_mirror.py`):

    {
      "pair_id":      str,
      "mirror_type":  "direction_flip" | "evidence_drop" | "family_swap"
                      | "abstain_unsafe",
      "prompt":       [ chat messages up through final user turn ],
      "chosen":       "<teacher assistant reply as string>",
      "rejected":     "<counterfactually perturbed assistant reply>",
      "prm_chosen":   float | null,
      "prm_rejected": float | null,    # always null for synthetic rejected
    }

Preference-construction strategies
----------------------------------
1.  **direction_flip** (primary)
    Source: full_correct / family_correct traces whose gold direction
    is directional (a_to_b or b_to_a).
    Rejected variant: take the chosen trace JSON, flip direction_tag in
    final_answer AND in any step whose role is "direction" or
    "conclusion".  Subject/object mentions in prose are left intact
    (this is a surface-level direction-tag hack -- we're teaching the
    model that THE TAG matters).
    Target: ~40 % of pairs.  Directly addresses the earlier baseline's 51 % mirror-error.

2.  **evidence_drop** (grounding)
    Source: full_correct traces with >=2 evidence_ids across steps.
    Rejected variant: clear all evidence_ids (replace with []) but
    keep rationale prose.  Teaches the student to prefer grounded
    reasoning.  This is a mild preference -- rationale prose without
    evidence_ids is WRONG under the schema, but fluent, so DPO alone
    won't nuke it.  Target: ~20 %.

3.  **family_swap** (classifier-level)
    Source: full_correct traces.
    Rejected: replace family with a plausible wrong family from a
    hand-built confusion set (e.g. PK_Metabolism <-> PD_Receptor) and
    replace subtype with an in-family subtype.  Teaches family
    discrimination beyond SFT token likelihood.  Target: ~20 %.

4.  **abstain_unsafe** (calibration)
    Source: near_miss traces (tier==near_miss).
    Rejected: original near_miss (wrong direction/polarity).
    Chosen: a synthesized abstention trace that pivots to
    role="evidence_gap" + explicit abstention and avoids the
    near_miss's structural error.  Teaches the student that honest
    abstention beats confidently-wrong.  Target: ~20 %.

Dedup & quality guards
----------------------
- Skip records where the final assistant message isn't valid JSON
  (schema drift shouldn't leak into preferences).
- Skip direction_flip where the gold is "bidirectional" (flipping is
  a no-op and would produce identical chosen/rejected).
- Skip evidence_drop where there are no evidence IDs to drop.
- Cap at `--max_per_pair` preferences per canonical pair_id so mirror
  augmentation doesn't over-index on a handful of examples.

Usage
-----
    python -m src.teacher.build_preference_pairs \
        --teacher_clean outputs/teacher/teacher_clean.jsonl \
        --prm_scores    outputs/teacher/prm_scores.jsonl \
        --output        outputs/preferences/mirror_pairs.jsonl \
        --report        outputs/audit/b6_preference_report.md \
        --strategies direction_flip,evidence_drop,family_swap,abstain_unsafe \
        --max_per_pair 2 \
        --seed 42
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.teacher.schema import (
    extract_json_block,
    VALID_DIRECTION_TAGS,
)


ROOT = Path(__file__).resolve().parents[2]

_FAMILY_CONFUSION: dict[str, list[str]] = {
    # Hand-built family-confusion map matching the actual family
    # vocabulary used in `data_processed/labels_hierarchical.parquet`.
    # Each family lists plausible cross-family confusions that a model
    # might emit; the `family_swap` strategy picks one as a hard
    # negative.
    "PK_Metabolism":   ["PK_Excretion",    "PK_Distribution"],
    "PK_Absorption":   ["PK_Distribution", "PK_Metabolism"],
    "PK_Distribution": ["PK_Metabolism",   "PK_Excretion"],
    "PK_Excretion":    ["PK_Metabolism",   "PK_Distribution"],
    "PD_Activity":     ["Efficacy",        "AdverseRisk"],
    "Efficacy":        ["PD_Activity",     "AdverseRisk"],
    "AdverseRisk":     ["PD_Activity",     "Efficacy"],
}

_IN_FAMILY_SUBTYPES: dict[str, list[str]] = {
    # Actual subtype vocabulary from the family taxonomy (see
    # configs/ddi_taxonomy.yaml / data_processed/taxonomy_schema.json).
    "PK_Metabolism":   ["metabolism", "serum_concentration",
                        "active_metabolite_serum_conc"],
    "PK_Excretion":    ["excretion_rate", "serum_concentration"],
    "PK_Absorption":   ["absorption_change", "bioavailability"],
    "PK_Distribution": ["protein_binding", "serum_concentration"],
    "PD_Activity":     ["activity_change",  "synergistic_effect",
                        "antagonistic_effect"],
    "Efficacy":        ["therapeutic_efficacy", "treatment_effectiveness"],
    "AdverseRisk":     ["adverse_effect_risk", "toxicity_risk",
                        "qt_prolongation_risk"],
}


def _parse_trace(record: dict) -> dict | None:
    """Extract and JSON-parse the teacher's assistant message."""
    msgs = record.get("messages") or []
    asst = next((m for m in msgs if m.get("role") == "assistant"), None)
    if asst is None:
        return None
    raw = asst.get("content") or ""
    try:
        parsed = extract_json_block(raw)
    except Exception:
        return None
    if not isinstance(parsed, dict) or "final_answer" not in parsed:
        return None
    return parsed


def _render_trace(trace: dict) -> str:
    """Render a trace dict back to the string format the assistant produced.

    We emit compact canonical JSON (sorted keys are NOT used because
    the teacher's original key order matters for stylistic fidelity).
    """
    return json.dumps(trace, ensure_ascii=False, indent=None, separators=(",", ":"))


def _flip_direction_tag(tag: str) -> str | None:
    if tag == "a_to_b":
        return "b_to_a"
    if tag == "b_to_a":
        return "a_to_b"
    return None  # bidirectional / n/a -> no flip


# ======================================================================
# Strategies
# ======================================================================
def _mk_direction_flip(rec: dict, trace: dict) -> dict | None:
    fa = trace.get("final_answer") or {}
    tag = fa.get("direction_tag")
    flipped = _flip_direction_tag(tag)
    if flipped is None:
        return None

    rejected_trace = copy.deepcopy(trace)
    rejected_trace["final_answer"]["direction_tag"] = flipped

    # Flip the tag in any per-step direction_tag field that points to the same.
    for step in rejected_trace.get("steps", []) or []:
        st_tag = step.get("direction_tag")
        if st_tag == tag:
            step["direction_tag"] = flipped

    return {
        "mirror_type": "direction_flip",
        "chosen":      _render_trace(trace),
        "rejected":    _render_trace(rejected_trace),
    }


def _mk_evidence_drop(rec: dict, trace: dict) -> dict | None:
    steps = trace.get("steps") or []
    total_eids = sum(len(s.get("evidence_ids") or []) for s in steps)
    if total_eids < 2:
        return None

    rejected_trace = copy.deepcopy(trace)
    for step in rejected_trace.get("steps", []) or []:
        step["evidence_ids"] = []

    return {
        "mirror_type": "evidence_drop",
        "chosen":      _render_trace(trace),
        "rejected":    _render_trace(rejected_trace),
    }


def _mk_family_swap(rec: dict, trace: dict, rng: random.Random) -> dict | None:
    fa = trace.get("final_answer") or {}
    fam = fa.get("family")
    if fam not in _FAMILY_CONFUSION:
        return None
    candidates = _FAMILY_CONFUSION[fam]
    wrong_fam = rng.choice(candidates)
    sub_candidates = _IN_FAMILY_SUBTYPES.get(wrong_fam, ["unknown_subtype"])
    wrong_sub = rng.choice(sub_candidates)

    rejected_trace = copy.deepcopy(trace)
    rejected_trace["final_answer"]["family"]  = wrong_fam
    rejected_trace["final_answer"]["subtype"] = wrong_sub
    # Also perturb the conclusion step's claim text surface
    for step in rejected_trace.get("steps", []) or []:
        if step.get("role") == "conclusion":
            # Simple, deterministic text substitution
            old_claim = step.get("claim") or ""
            new_claim = old_claim.replace(fam, wrong_fam)
            step["claim"] = new_claim if new_claim != old_claim else (
                f"The pair's interaction is best characterized as {wrong_fam}/{wrong_sub}."
            )
    return {
        "mirror_type": "family_swap",
        "chosen":      _render_trace(trace),
        "rejected":    _render_trace(rejected_trace),
    }


def _mk_abstain_unsafe(rec: dict, trace: dict) -> dict | None:
    """For tier==near_miss, build a pair where the original (wrong-commit)
    trace is *rejected* and a synthesized abstention is *chosen*.

    The synthesized chosen preserves the evidence-gathering steps (role
    in {pathway, protein, pk_flag, mechanism_of_action, evidence_gap})
    and replaces the conclusion with an abstention step + a conservative
    final_answer (family="abstain", direction_tag="n/a", abstain=True).
    """
    if rec.get("tier") != "near_miss":
        return None

    safe_trace = copy.deepcopy(trace)
    kept_steps = []
    for step in safe_trace.get("steps", []) or []:
        if step.get("role") == "conclusion":
            continue
        kept_steps.append(step)
    # Add a crisp abstention step.
    abst_id = max((s.get("step_id", 0) for s in kept_steps), default=0) + 1
    kept_steps.append({
        "step_id":        abst_id,
        "role":           "evidence_gap",
        "claim":          "The available retrieval evidence does not uniquely "
                          "determine the direction or subtype; declining to "
                          "commit is safer than a confident mis-prediction.",
        "evidence_ids":   [],
        "direction_tag":  "n/a",
    })
    kept_steps.append({
        "step_id":        abst_id + 1,
        "role":           "abstention",
        "claim":          "Abstaining on this pair.",
        "evidence_ids":   [],
        "direction_tag":  "n/a",
    })
    safe_trace["steps"] = kept_steps
    safe_trace["final_answer"] = {
        "family":        "abstain",
        "subtype":       "abstain",
        "direction_tag": "n/a",
        "polarity":      None,
        "abstain":       True,
        "confidence":    0.0,
    }

    return {
        "mirror_type": "abstain_unsafe",
        "chosen":      _render_trace(safe_trace),
        "rejected":    _render_trace(trace),
    }


_STRATEGY_IMPL = {
    "direction_flip": lambda rec, tr, rng: _mk_direction_flip(rec, tr),
    "evidence_drop":  lambda rec, tr, rng: _mk_evidence_drop(rec, tr),
    "family_swap":    lambda rec, tr, rng: _mk_family_swap(rec, tr, rng),
    "abstain_unsafe": lambda rec, tr, rng: _mk_abstain_unsafe(rec, tr),
}


# ======================================================================
# PRM score join
# ======================================================================
def _load_prm_scores(path: Path | None) -> dict[str, float]:
    """Load trace-level PRM mean scores keyed by pair_id."""
    if path is None or not path.exists():
        return {}
    out: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("pair_id")
            if pid is None:
                continue
            # Accept several shapes:  {"prm_mean": 0.8} OR {"steps": [...]}
            if "prm_mean" in r:
                out[pid] = float(r["prm_mean"])
            elif "steps" in r and isinstance(r["steps"], list):
                scores = [float(s.get("prm_score", 0.0)) for s in r["steps"]
                          if isinstance(s.get("prm_score", None), (int, float))]
                if scores:
                    out[pid] = sum(scores) / len(scores)
    return out


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_clean", required=True)
    ap.add_argument("--prm_scores",   default=None,
                    help="Optional PRM score JSONL; used for prm_chosen.")
    ap.add_argument("--output",       required=True)
    ap.add_argument("--report",       default=None)
    ap.add_argument("--strategies",   default="direction_flip,evidence_drop,family_swap,abstain_unsafe")
    ap.add_argument("--max_per_pair", type=int, default=2)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--include_tiers", default="full_correct,family_correct,near_miss",
                    help="Comma-separated tiers to draw chosen traces from.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    include_tiers = {t.strip() for t in args.include_tiers.split(",") if t.strip()}

    prm = _load_prm_scores(Path(args.prm_scores)) if args.prm_scores else {}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hist: Counter = Counter()
    skipped: Counter = Counter()
    n_in = n_out = 0

    with open(args.teacher_clean) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_in += 1

            tier = rec.get("tier")
            if tier not in include_tiers:
                skipped[f"tier_excluded:{tier}"] += 1
                continue

            trace = _parse_trace(rec)
            if trace is None:
                skipped["bad_json"] += 1
                continue

            # Prompt = all messages except the assistant's final reply
            prompt_msgs = [m for m in rec.get("messages", [])
                           if m.get("role") != "assistant"]
            pair_id = rec.get("pair_id") or rec.get("pair_key") or ""
            prm_chosen = prm.get(pair_id)

            per_pair_emitted = 0
            # Randomize strategy order so --max_per_pair picks a diverse set
            strat_order = strategies[:]
            rng.shuffle(strat_order)
            for strat in strat_order:
                if per_pair_emitted >= args.max_per_pair:
                    break
                impl = _STRATEGY_IMPL.get(strat)
                if impl is None:
                    skipped[f"unknown_strategy:{strat}"] += 1
                    continue
                out = impl(rec, trace, rng)
                if out is None:
                    skipped[f"strategy_na:{strat}"] += 1
                    continue
                pref = {
                    "pair_id":      pair_id,
                    "mirror_type":  out["mirror_type"],
                    "prompt":       prompt_msgs,
                    "chosen":       out["chosen"],
                    "rejected":     out["rejected"],
                    "prm_chosen":   prm_chosen,
                    "prm_rejected": None,
                    "source_tier":  tier,
                }
                fout.write(json.dumps(pref, ensure_ascii=False) + "\n")
                hist[out["mirror_type"]] += 1
                per_pair_emitted += 1
                n_out += 1

    print(f"[b6] read {n_in:,} teacher records; emitted {n_out:,} preference pairs")
    print(f"[b6] by mirror_type: {dict(hist)}")
    print(f"[b6] skipped: {dict(skipped)}")

    if args.report:
        rep_path = Path(args.report)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_path, "w") as f:
            f.write("# B6 -- Mirror preference-pair report\n\n")
            f.write(f"- input (`teacher_clean`): `{args.teacher_clean}`\n")
            f.write(f"- output                  : `{args.output}`\n")
            f.write(f"- records read            : {n_in:,}\n")
            f.write(f"- preference pairs emitted: {n_out:,}\n\n")
            f.write("## By mirror type\n\n| type | n |\n|---|---:|\n")
            for k, v in sorted(hist.items()):
                f.write(f"| `{k}` | {v:,} |\n")
            if skipped:
                f.write("\n## Skipped reasons\n\n| reason | n |\n|---|---:|\n")
                for k, v in sorted(skipped.items()):
                    f.write(f"| `{k}` | {v:,} |\n")
        print(f"[b6] wrote report {rep_path}")


if __name__ == "__main__":
    main()
