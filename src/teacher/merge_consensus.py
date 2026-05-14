"""B5-v2 -- Cross-teacher consensus merge for V4.

The original `src/teacher/merge.py` collapses winners across the three
teacher critic files by `pair_id`, keeping the highest critic score and
discarding the other two predictions entirely. That throws away ~67% of
the cross-teacher signal we paid for. This script keeps it.

For each pair_id we:
  1.  Gather up to N per-teacher winner candidates (typically 3).
  2.  Compute consensus statistics over their non-abstain predictions:
        k_family     -- max number of teachers that agree on a family
        k_subtype    -- same, on (family, subtype)
        k_direction  -- same, on direction_tag
        majority_*   -- the agreed-upon value (None if no plurality)
  3.  Pick the canonical winner with this priority order:
        a.  PRM final_plus >= min_score AND tier != 'drop' AND
            predicted_family == gold_family
        b.  Tier rank (full_correct > family_correct > near_miss > abstention)
        c.  PRM `final_plus` (the trace-level / ORM-style head; supervised
            on critical_passed)
        d.  PRM `geomean_plus` (tie-break)
  4.  Compute a consensus-aware sample weight:
         w = tier_weight
             * (1 + boost_family * (k_family - 1)
                + boost_subtype * (k_subtype - 1)
                + boost_direction * (k_direction - 1))
             * (disputed_penalty if no plurality on family else 1.0)
      Capped at `--max_weight` so a single high-consensus pair cannot
      dominate the SFT loss.
  5.  Emit the SFT record (identical schema to the v1 merge), plus a
      sidecar consensus_audit.jsonl with the per-teacher predictions.

Note on PRM scores
------------------
Earlier versions of this script used `min_plus` (min over per-step P(+))
as the trace-level PRM score.  Empirically, our PRM was trained step-level
on a strict-tier-heavy corpus, which causes per-step P(+) to saturate near
0 for most intermediate steps.  `min_plus` and `geomean_plus` therefore
collapse to the noise floor and provide ~no ranking signal.  The PRM's
final-step P(+) is supervised on `critical_passed` and behaves like an
ORM head with healthy spread — it is the score we now use everywhere.
The `--min_score` default has dropped accordingly (final_plus is unitless
but is calibrated against critical_passed, not raw step soundness).

Why this matters for V4:
  - V3's "correct reasoning, wrong class" failure looks like exactly the
    case where 2/3 teachers agree on the right family but the gold subtype
    sits in the long tail. Consensus weighting boosts those examples
    *without* dropping the third teacher's signal.
  - Disputed cases (k_family == 1) are inherently noisier. Downweighting
    prevents them from dragging the loss; we don't drop them because they
    contain useful diversity for the synthetic-expansion stage.

Usage:
    python -m src.teacher.merge_consensus \\
        --critic outputs/teacher/critic_<TEACHER1>.reranked.jsonl \\
                 outputs/teacher/critic_<TEACHER2>.reranked.jsonl \\
                 outputs/teacher/critic_<TEACHER3>.reranked.jsonl \\
        --output outputs/teacher/teacher_clean.consensus.jsonl \\
        --audit  outputs/teacher/consensus_audit.jsonl \\
        --split  subset25k \\
        --min_score 0.0 \\
        --max_weight 2.5
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from src.teacher.context_builder import ContextBuilder, DATA
from src.teacher.prompt import build_prompt

ROOT = Path(__file__).resolve().parents[2]


# Same tier weights as src/teacher/merge.py for compatibility with the
# rest of Phase C. Consensus boost is multiplicative on top.
TIER_WEIGHTS = {
    "full_correct":   1.00,
    "family_correct": 0.50,
    "abstention":     0.33,
    "near_miss":      0.25,
}

# Tier rank for the winner-selection tie-break (higher = preferred).
TIER_RANK = {
    "full_correct":   3,
    "family_correct": 2,
    "near_miss":      1,
    "abstention":     0,
    "drop":          -1,
}

# Default consensus boost coefficients. Family agreement matters most
# because family is the headline classification target; subtype/direction
# are secondary refinements. Tuned to give k_family=3 a ~2x weight at
# tier_weight=1, before the max_weight cap.
DEFAULT_BOOST = {"family": 0.5, "subtype": 0.3, "direction": 0.2}

# When no two teachers agree on a family, downweight the SFT signal but
# do not drop the pair (we still want the diverse training example).
DEFAULT_DISPUTED_PENALTY = 0.5


def _teacher_id_from_path(path: Path) -> str:
    name = path.stem
    for prefix in ("critic_", "qc_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def _final_pred(parsed: dict | None) -> dict | None:
    """Project a parsed assistant trace to (family, subtype, direction, abstain)."""
    if not isinstance(parsed, dict):
        return None
    fa = parsed.get("final_answer")
    if not isinstance(fa, dict):
        return None
    return {
        "family":        fa.get("family"),
        "subtype":       fa.get("subtype"),
        "direction_tag": fa.get("direction_tag"),
        "polarity":      fa.get("polarity"),
        "abstain":       bool(fa.get("abstain", False)),
    }


def _committed(pred: dict | None) -> bool:
    return bool(pred and not pred["abstain"]
                and pred.get("family") and pred.get("subtype"))


def _consensus(preds: list[dict]) -> dict:
    """Compute agreement counts over committed predictions only."""
    fam_counts: Counter[str] = Counter()
    sub_counts: Counter[tuple] = Counter()
    dir_counts: Counter[str] = Counter()
    for p in preds:
        if not _committed(p):
            continue
        fam_counts[p["family"]] += 1
        sub_counts[(p["family"], p["subtype"])] += 1
        if p["direction_tag"]:
            dir_counts[p["direction_tag"]] += 1

    def _top(counter):
        if not counter:
            return None, 0
        item, n = counter.most_common(1)[0]
        return item, n

    fam_majority, k_family = _top(fam_counts)
    (sub_fam, sub_st), k_subtype = (
        (None, None), 0
    ) if not sub_counts else (sub_counts.most_common(1)[0][0],
                              sub_counts.most_common(1)[0][1])
    dir_majority, k_direction = _top(dir_counts)

    return {
        "n_voters":      sum(1 for p in preds if _committed(p)),
        "k_family":      int(k_family),
        "k_subtype":     int(k_subtype),
        "k_direction":   int(k_direction),
        "majority_family":    fam_majority,
        "majority_subtype":   None if sub_fam is None else f"{sub_fam}/{sub_st}",
        "majority_direction": dir_majority,
        "family_votes":  dict(fam_counts),
        "subtype_votes": {f"{f}/{s}": n for (f, s), n in sub_counts.items()},
        "direction_votes": dict(dir_counts),
    }


def _select_winner(records_by_teacher: dict[str, dict],
                   gold_family: str | None,
                   min_score: float) -> tuple[str | None, dict | None]:
    """Pick the winning (teacher, record) tuple per the priority order in
    the module docstring. Returns (None, None) if everyone failed."""
    candidates = []
    for tid, w in records_by_teacher.items():
        wc = w["winner_candidate"]
        # final_plus is the PRM's ORM-style head (supervised on
        # critical_passed) — the only step aggregation that is not
        # saturated for our strict-tier-heavy PRM.  geomean_plus is kept
        # as a tertiary tie-break for cases where final_plus is exactly
        # equal across teachers.
        score = w["score"].get("final_plus", 0.0)
        score_alt = w["score"].get("geomean_plus", 0.0)
        tier = wc.get("tier", "drop")
        pred = _final_pred(wc.get("parsed"))
        family_match = (pred is not None
                        and pred.get("family") == gold_family
                        and not pred.get("abstain"))
        candidates.append({
            "teacher": tid,
            "rec": w,
            "tier": tier,
            "score": score,
            "score_alt": score_alt,
            "family_match": family_match,
        })

    # Filter: anything below min_score or tier=='drop' is unfit.  With the
    # default min_score=0.0 this is effectively just the tier filter.
    fit = [c for c in candidates
           if c["score"] >= min_score and c["tier"] != "drop"]
    if not fit:
        return None, None

    # Order: family_match desc, tier_rank desc, final_plus desc, geomean_plus desc
    fit.sort(key=lambda c: (
        c["family_match"],
        TIER_RANK.get(c["tier"], -1),
        c["score"],
        c["score_alt"],
    ), reverse=True)
    win = fit[0]
    return win["teacher"], win["rec"]


def _consensus_weight(base: float, cons: dict, boost: dict,
                      disputed_penalty: float, max_weight: float) -> float:
    if cons["n_voters"] == 0:
        return base
    factor = 1.0
    if cons["k_family"] >= 1:
        factor += boost["family"] * (cons["k_family"] - 1)
    if cons["k_subtype"] >= 1:
        factor += boost["subtype"] * (cons["k_subtype"] - 1)
    if cons["k_direction"] >= 1:
        factor += boost["direction"] * (cons["k_direction"] - 1)
    # Disputed = at least 2 voters but no plurality on family
    if cons["n_voters"] >= 2 and cons["k_family"] <= 1:
        factor *= disputed_penalty
    return min(base * factor, max_weight)


def merge(critic_paths: list[Path], split: str, out_path: Path,
          audit_path: Path, min_score: float = 0.60,
          max_weight: float = 2.5,
          boost: dict | None = None,
          disputed_penalty: float = DEFAULT_DISPUTED_PENALTY,
          require_committed: bool = False) -> dict:
    boost = {**DEFAULT_BOOST, **(boost or {})}
    cb = ContextBuilder(neighbor_index_path=DATA / f"neighbor_index_{split}.parquet")

    # Group winners by pair_id, keyed by teacher id
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for cp in critic_paths:
        tid = _teacher_id_from_path(cp)
        with cp.open() as f:
            for line in f:
                w = json.loads(line)
                pid = w["pair_id"]
                if tid in by_pair[pid]:
                    # Two records for same pair from same teacher: keep
                    # the higher final_plus (deterministic).
                    incoming = w["score"].get("final_plus", 0.0)
                    existing = by_pair[pid][tid]["score"].get("final_plus", 0.0)
                    if incoming <= existing:
                        continue
                by_pair[pid][tid] = w

    n_pairs = len(by_pair)
    print(f"[merge_consensus] {n_pairs:,} unique pair_ids across "
          f"{len(critic_paths)} critic files")

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_emitted = 0
    n_dropped_no_winner = 0
    n_dropped_uncommitted = 0
    fam_dist: Counter[str] = Counter()
    tier_dist: Counter[str] = Counter()
    consensus_dist: Counter[int] = Counter()  # k_family histogram
    weight_sum = 0.0

    with out_path.open("w") as fout, audit_path.open("w") as faud:
        for pid, by_teacher in by_pair.items():
            preds = [_final_pred(w["winner_candidate"].get("parsed"))
                     for w in by_teacher.values()]
            preds = [p for p in preds if p is not None]
            cons = _consensus(preds)

            # Gold from the first winner (all teachers see the same gold).
            gold_family = next(iter(by_teacher.values()))[
                "winner_candidate"].get("gold_family")

            tid, w = _select_winner(by_teacher, gold_family, min_score)
            if tid is None or w is None:
                n_dropped_no_winner += 1
                faud.write(json.dumps({
                    "pair_id": pid,
                    "drop_reason": "no_fit_winner",
                    "consensus": cons,
                    "teacher_predictions": {
                        t: _final_pred(rw["winner_candidate"].get("parsed"))
                        for t, rw in by_teacher.items()
                    },
                }) + "\n")
                continue

            wc = w["winner_candidate"]
            tier = wc.get("tier", "drop")
            parsed = wc.get("parsed")
            if parsed is None:
                n_dropped_no_winner += 1
                continue
            pred = _final_pred(parsed) or {}
            if require_committed and pred.get("abstain"):
                n_dropped_uncommitted += 1
                continue

            base_w = TIER_WEIGHTS.get(tier, 0.25)
            sample_weight = _consensus_weight(
                base_w, cons, boost, disputed_penalty, max_weight,
            )

            ctx = cb.build(pid)
            msgs = build_prompt(ctx)
            assistant_json = json.dumps(parsed, separators=(",", ":"))

            rec = {
                "pair_id": pid,
                "family": wc.get("gold_family"),
                "subtype": wc.get("gold_subtype"),
                "direction_tag": wc.get("gold_direction"),
                "polarity": wc.get("gold_polarity"),
                "messages": [
                    {"role": "system", "content": msgs["system"]},
                    {"role": "user",   "content": msgs["user"]},
                    {"role": "assistant", "content": assistant_json},
                ],
                "critic_score": w["score"].get("final_plus", 0.0),
                "critic_score_breakdown": {
                    "final_plus":   w["score"].get("final_plus"),
                    "geomean_plus": w["score"].get("geomean_plus"),
                    "mean_plus":    w["score"].get("mean_plus"),
                    "min_plus":     w["score"].get("min_plus"),
                },
                "qc_strict_passed":   bool(wc.get("passed")),
                "qc_critical_passed": bool(wc.get("critical_passed")),
                "tier": tier,
                "sample_weight": sample_weight,
                "n_candidates": w.get("n_candidates", 1),
                # New v2 fields:
                "winning_teacher": tid,
                "consensus": cons,
                "context_ids": sorted(ctx.context_ids()),
            }
            fout.write(json.dumps(rec) + "\n")

            faud.write(json.dumps({
                "pair_id": pid,
                "winning_teacher": tid,
                "consensus": cons,
                "sample_weight": sample_weight,
                "tier": tier,
                "teacher_predictions": {
                    t: _final_pred(rw["winner_candidate"].get("parsed"))
                    for t, rw in by_teacher.items()
                },
                "teacher_scores": {
                    t: rw["score"].get("final_plus", 0.0)
                    for t, rw in by_teacher.items()
                },
                "teacher_tiers": {
                    t: rw["winner_candidate"].get("tier")
                    for t, rw in by_teacher.items()
                },
            }) + "\n")

            n_emitted += 1
            fam_dist[wc.get("gold_family") or "?"] += 1
            tier_dist[tier] += 1
            consensus_dist[cons["k_family"]] += 1
            weight_sum += sample_weight

    # Audit summary
    avg_w = weight_sum / max(1, n_emitted)
    audit_md = ROOT / "outputs" / "audit" / "b5_consensus_merge_report.md"
    audit_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# B5 (v2) consensus-merge report",
        "",
        f"- Critic inputs: {len(critic_paths)} file(s), "
        f"{', '.join(p.name for p in critic_paths)}",
        f"- Unique pair_ids in any critic file: {n_pairs:,}",
        f"- Pair_ids emitted to `{out_path}`: {n_emitted:,}",
        f"- Pair_ids dropped (no fit winner / score < {min_score} or all-drop): "
        f"{n_dropped_no_winner:,}",
        f"- Pair_ids dropped (uncommitted, --require_committed): "
        f"{n_dropped_uncommitted:,}",
        f"- Average sample_weight: {avg_w:.3f}",
        f"- max_weight cap: {max_weight}",
        "",
        "## k_family histogram (post-emit)",
        "",
        "| k_family | count | share |",
        "|----------|-------|-------|",
    ]
    for k in sorted(consensus_dist):
        c = consensus_dist[k]
        lines.append(f"| {k} | {c:,} | {100*c/max(1,n_emitted):.1f}% |")

    lines += [
        "",
        "## Tier distribution",
        "",
        "| Tier | Count | Share | Tier weight |",
        "|------|-------|-------|-------------|",
    ]
    for t in ("full_correct", "family_correct", "near_miss", "abstention"):
        c = tier_dist.get(t, 0)
        lines.append(
            f"| {t} | {c:,} | {100*c/max(1,n_emitted):.1f}% | "
            f"{TIER_WEIGHTS[t]:.2f} |"
        )

    lines += [
        "",
        "## Family distribution",
        "",
        "| Family | Count | Share |",
        "|--------|-------|-------|",
    ]
    for fam, c in sorted(fam_dist.items(), key=lambda x: -x[1]):
        lines.append(f"| {fam} | {c:,} | {100*c/max(1,n_emitted):.1f}% |")

    audit_md.write_text("\n".join(lines) + "\n")
    print(f"[merge_consensus] wrote {audit_md.relative_to(ROOT)}")
    print(f"[merge_consensus] wrote {out_path}")
    print(f"[merge_consensus] wrote {audit_path}")
    print(f"[merge_consensus] emitted={n_emitted:,} avg_weight={avg_w:.3f}")
    print(f"[merge_consensus] k_family histogram: "
          f"{dict(sorted(consensus_dist.items()))}")
    return {
        "n_emitted":      n_emitted,
        "n_dropped":      n_dropped_no_winner + n_dropped_uncommitted,
        "k_family_hist":  dict(consensus_dist),
        "tier_dist":      dict(tier_dist),
        "family_dist":    dict(fam_dist),
        "avg_weight":     avg_w,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--critic", nargs="+", required=True,
                   help="One or more critic_*.jsonl files (one per teacher).")
    p.add_argument("--output", required=True,
                   help="teacher_clean.consensus.jsonl path.")
    p.add_argument("--audit", default=None,
                   help="Path to per-pair consensus audit JSONL "
                        "(default: <output>.audit.jsonl).")
    p.add_argument("--split", default="subset25k")
    p.add_argument("--min_score", type=float, default=0.0,
                   help="Minimum PRM final_plus to admit a teacher's winner. "
                        "Default 0.0 means we rely entirely on the tier "
                        "filter (tier!='drop').  Increase (e.g. 0.3) for a "
                        "stricter, smaller corpus.")
    p.add_argument("--max_weight", type=float, default=2.5,
                   help="Cap on the consensus-boosted sample weight.")
    p.add_argument("--boost_family", type=float, default=DEFAULT_BOOST["family"])
    p.add_argument("--boost_subtype", type=float, default=DEFAULT_BOOST["subtype"])
    p.add_argument("--boost_direction", type=float, default=DEFAULT_BOOST["direction"])
    p.add_argument("--disputed_penalty", type=float,
                   default=DEFAULT_DISPUTED_PENALTY,
                   help="Multiplier on sample_weight when no two teachers "
                        "agree on a family.")
    p.add_argument("--require_committed", action="store_true",
                   help="Drop the pair if the chosen winner is an abstention "
                        "(useful for the commits-only ablation).")
    args = p.parse_args()

    audit = Path(args.audit) if args.audit else Path(args.output + ".audit.jsonl")
    merge(
        critic_paths=[Path(p) for p in args.critic],
        split=args.split,
        out_path=Path(args.output),
        audit_path=audit,
        min_score=args.min_score,
        max_weight=args.max_weight,
        boost={
            "family":    args.boost_family,
            "subtype":   args.boost_subtype,
            "direction": args.boost_direction,
        },
        disputed_penalty=args.disputed_penalty,
        require_committed=args.require_committed,
    )


if __name__ == "__main__":
    main()
