"""Build Phase 4 hard-negative preferences from the targeted repair corpus.

Chosen examples are high-quality teacher repair traces. Rejected examples are
synthetic but schema-valid perturbations that mimic the observed student
failure modes:

  - family_swap_to_adverse: rare PK/PD/Efficacy family collapsed to AdverseRisk
  - family_axis_swap: PK_* family confused with another PK axis
  - subtype_swap: correct family, wrong in-family subtype
  - direction_flip: correct family/subtype, wrong direction tag

This is intentionally narrower than the older build_preference_pairs.py:
all rejected subtype values come from the actual prompt whitelist, and no
near_miss/abstention examples are included by default.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.teacher.prompt import SUBTYPE_VOCAB


TARGET_FAMILIES = {
    "PK_Metabolism",
    "PK_Excretion",
    "PK_Distribution",
    "PK_Absorption",
    "PD_Activity",
    "Efficacy",
}

FAMILY_AXIS_CONFUSIONS = {
    "PK_Metabolism": ["AdverseRisk", "PK_Excretion", "PK_Distribution"],
    "PK_Excretion": ["AdverseRisk", "PK_Metabolism", "PK_Distribution"],
    "PK_Distribution": ["AdverseRisk", "PK_Metabolism", "PK_Excretion"],
    "PK_Absorption": ["AdverseRisk", "PK_Distribution", "PK_Metabolism"],
    "PD_Activity": ["AdverseRisk", "Efficacy"],
    "Efficacy": ["AdverseRisk", "PD_Activity"],
    "AdverseRisk": ["PD_Activity", "Efficacy"],
}

ADVERSE_SUBTYPE_FOR_SOURCE = {
    "PK_Metabolism": "adverse_effects",
    "PK_Excretion": "adverse_effects",
    "PK_Distribution": "adverse_effects",
    "PK_Absorption": "adverse_effects",
    "PD_Activity": "adverse_effects",
    "Efficacy": "adverse_effects",
}

DIR_FLIP = {"a_to_b": "b_to_a", "b_to_a": "a_to_b"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _assistant_content(rec: dict[str, Any]) -> str | None:
    for msg in rec.get("messages") or []:
        if msg.get("role") == "assistant":
            return msg.get("content") or ""
    return None


def _prompt_messages(rec: dict[str, Any]) -> list[dict[str, str]]:
    return [m for m in rec.get("messages") or [] if m.get("role") != "assistant"]


def _render(trace: dict[str, Any]) -> str:
    return json.dumps(trace, ensure_ascii=False, separators=(",", ":"))


def _parse_trace(rec: dict[str, Any]) -> dict[str, Any] | None:
    raw = _assistant_content(rec)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("final_answer"), dict):
        return None
    return obj


def _set_family(trace: dict[str, Any], family: str, subtype: str) -> dict[str, Any]:
    out = copy.deepcopy(trace)
    out["final_answer"]["family"] = family
    out["final_answer"]["subtype"] = subtype
    for step in out.get("steps") or []:
        if step.get("role") == "conclusion":
            step["claim"] = (
                f"The pair's interaction is best characterized as "
                f"{family}/{subtype}."
            )
        if step.get("family_hint") in TARGET_FAMILIES | {"AdverseRisk"}:
            step["family_hint"] = family
    return out


def _swap_subtype(trace: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    fam = (trace.get("final_answer") or {}).get("family")
    cur = (trace.get("final_answer") or {}).get("subtype")
    choices = [s for s in SUBTYPE_VOCAB.get(fam, []) if s != cur]
    if not choices:
        return None
    subtype = rng.choice(choices)
    out = copy.deepcopy(trace)
    out["final_answer"]["subtype"] = subtype
    for step in out.get("steps") or []:
        if step.get("role") == "conclusion":
            step["claim"] = (
                f"The pair's interaction has subtype {subtype}."
            )
    return out


def _flip_direction(trace: dict[str, Any]) -> dict[str, Any] | None:
    tag = (trace.get("final_answer") or {}).get("direction_tag")
    flipped = DIR_FLIP.get(tag)
    if not flipped:
        return None
    out = copy.deepcopy(trace)
    out["final_answer"]["direction_tag"] = flipped
    for step in out.get("steps") or []:
        st = step.get("direction_tag")
        if st in DIR_FLIP:
            step["direction_tag"] = DIR_FLIP[st]
    return out


def _mk_pref(rec: dict[str, Any], trace: dict[str, Any],
             rejected: dict[str, Any], kind: str) -> dict[str, Any]:
    prm = rec.get("critic_score")
    if prm is None:
        prm = (rec.get("critic_score_breakdown") or {}).get("final_plus")
    return {
        "pair_id": rec.get("pair_id") or "",
        "mirror_type": kind,
        "prompt": _prompt_messages(rec),
        "chosen": _render(trace),
        "rejected": _render(rejected),
        "prm_chosen": float(prm) if prm is not None else None,
        "prm_rejected": None,
        "source_family": rec.get("family"),
        "source_tier": rec.get("tier"),
        "source_quality": rec.get("quality_score"),
        "source_weight": rec.get("sample_weight"),
    }


def build(input_path: Path, output_path: Path, report_path: Path,
          include_families: set[str], include_tiers: set[str],
          max_per_pair: int, max_total: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = _load_jsonl(input_path)
    prefs: list[dict[str, Any]] = []
    stats: Counter = Counter()
    by_pair: Counter = Counter()

    # Prefer families/records that still need polish.
    def priority(rec: dict[str, Any]) -> tuple:
        fam = rec.get("family") or ""
        fam_rank = {
            "PK_Distribution": 6,
            "PK_Metabolism": 5,
            "Efficacy": 4,
            "PD_Activity": 3,
            "PK_Excretion": 2,
            "PK_Absorption": 1,
            "AdverseRisk": 0,
        }.get(fam, 0)
        k_family = (rec.get("consensus") or {}).get("k_family") or 1
        qs = float(rec.get("quality_score") or 0)
        return (fam_rank, k_family, qs)

    for rec in sorted(rows, key=priority, reverse=True):
        stats["input"] += 1
        pid = rec.get("pair_id") or ""
        fam = rec.get("family")
        tier = rec.get("tier")
        if fam not in include_families:
            stats[f"skip_family:{fam}"] += 1
            continue
        if tier not in include_tiers:
            stats[f"skip_tier:{tier}"] += 1
            continue
        if by_pair[pid] >= max_per_pair:
            stats["skip_max_per_pair"] += 1
            continue
        trace = _parse_trace(rec)
        if trace is None:
            stats["skip_bad_trace"] += 1
            continue

        candidates: list[tuple[str, dict[str, Any] | None]] = []
        if fam != "AdverseRisk":
            adv_sub = ADVERSE_SUBTYPE_FOR_SOURCE.get(fam, "adverse_effects")
            candidates.append((
                "family_swap_to_adverse",
                _set_family(trace, "AdverseRisk", adv_sub),
            ))
        for wrong_fam in FAMILY_AXIS_CONFUSIONS.get(fam, []):
            if wrong_fam == "AdverseRisk":
                continue
            subtype = SUBTYPE_VOCAB.get(wrong_fam, ["adverse_effects"])[0]
            candidates.append((
                "family_axis_swap",
                _set_family(trace, wrong_fam, subtype),
            ))
        candidates.append(("subtype_swap", _swap_subtype(trace, rng)))
        candidates.append(("direction_flip", _flip_direction(trace)))

        for kind, rejected in candidates:
            if rejected is None:
                stats[f"skip_strategy:{kind}"] += 1
                continue
            if by_pair[pid] >= max_per_pair:
                break
            prefs.append(_mk_pref(rec, trace, rejected, kind))
            by_pair[pid] += 1
            stats["output"] += 1
            stats[f"output_type:{kind}"] += 1
            stats[f"output_family:{fam}"] += 1
            if max_total > 0 and len(prefs) >= max_total:
                break
        if max_total > 0 and len(prefs) >= max_total:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for pref in prefs:
            f.write(json.dumps(pref, ensure_ascii=False) + "\n")

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "n_preferences": len(prefs),
        "n_pair_ids": len({p["pair_id"] for p in prefs}),
        "include_families": sorted(include_families),
        "include_tiers": sorted(include_tiers),
        "max_per_pair": max_per_pair,
        "max_total": max_total,
        "stats": dict(stats),
        "preferences_by_pair_hist": dict(Counter(by_pair.values())),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    p.add_argument(
        "--include_families",
        default="PK_Distribution,PK_Metabolism,Efficacy,PD_Activity,PK_Excretion,PK_Absorption",
    )
    p.add_argument("--include_tiers", default="full_correct,family_correct")
    p.add_argument("--max_per_pair", type=int, default=3)
    p.add_argument("--max_total", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    include_families = {
        x.strip() for x in args.include_families.split(",") if x.strip()
    }
    include_tiers = {
        x.strip() for x in args.include_tiers.split(",") if x.strip()
    }
    report = build(
        Path(args.input),
        Path(args.output),
        Path(args.report),
        include_families=include_families,
        include_tiers=include_tiers,
        max_per_pair=args.max_per_pair,
        max_total=args.max_total,
        seed=args.seed,
    )
    print(f"[phase4_prefs] wrote {args.output}")
    print(f"[phase4_prefs] report {args.report}")
    print(f"[phase4_prefs] n_preferences={report['n_preferences']} "
          f"n_pair_ids={report['n_pair_ids']}")
    for k, v in sorted(report["stats"].items()):
        print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
