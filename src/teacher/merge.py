"""B5 — Merge winner traces into teacher_clean.jsonl.

Consumes:
  outputs/teacher/critic_subset25k*.jsonl   (B4 winners, one per pair)

Emits:
  outputs/teacher/teacher_clean.jsonl       (one record per pair; the SFT set)
  outputs/audit/b5_merge_report.md

The SFT record schema:
  {
    "pair_id":          str,
    "family":           str,          # gold family (verified)
    "subtype":          str,
    "direction_tag":    str,
    "polarity":         str | null,
    "messages": [                     # chat-format for SFT trainer
        {"role": "system", "content": "<teacher-system>"},
        {"role": "user",   "content": "<teacher-user>"},
        {"role": "assistant", "content": "<chosen candidate JSON>"}
    ],
    "critic_score":     float,        # PRM min_plus of the chosen candidate
    "qc_strict_passed": bool,
    "tier":             str,          # full_correct | family_correct | near_miss
    "sample_weight":    float,        # 1.0 / 0.5 / 0.25 by tier (Phase C)
    "n_candidates":     int
  }

Tier-based SFT weighting (near-miss preservation fix):
  - full_correct   → weight 1.00   (subtype exact, all critical gates pass)
  - family_correct → weight 0.50   (family+direction right, subtype in whitelist)
  - near_miss      → weight 0.25   (family right, subtype in whitelist, but
                                     direction or polarity soft-fails)

This addresses the "correct reasoning, wrong class" failure mode:
the student gets exposed to teachers with correct mechanistic reasoning
even when the exact subtype disagrees with the DrugBank label.

Dedup + stratification:
  - One record per pair_id  (winner only).
  - Log family-balance before/after so the paper can report it.
  - Also emit a mirror-pair index (for C3 MPS consistency loss): every pair
    that has a mirror in teacher_clean gets a `mirror_pair_id` entry in a
    separate sidecar file.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from src.teacher.schema import extract_json_block
from src.teacher.context_builder import ContextBuilder, DATA
from src.teacher.prompt import build_prompt

ROOT = Path(__file__).resolve().parents[2]


# Phase C SFT sample weights per tier.
# Rationale:
#   full_correct (1.00): committed + family/subtype/polarity all right -- gold
#   family_correct (0.50): committed + family right, subtype in whitelist
#   near_miss (0.25): family right but direction/polarity wrong (structural issue)
#   abstention (0.33): honest uncertainty with intact schema + grounded evidence.
#     Slightly above near_miss because abstention teaches calibrated uncertainty
#     (core goal), slightly below family_correct because the density of
#     training signal per token is lower than a confident correct answer.
TIER_WEIGHTS = {
    "full_correct":   1.00,
    "family_correct": 0.50,
    "abstention":     0.33,
    "near_miss":      0.25,
}


def merge(critic_paths: list[Path], split: str, out_path: Path,
          min_score: float = 0.60, require_strict: bool = True,
          keep_near_miss: bool = True):
    """Merge critic winners into teacher_clean with tier-based weighting.

    keep_near_miss=True (default) preserves family_correct and
    near_miss tiers with down-weighted sample weights so Phase C student
    SFT can still learn from traces with good reasoning but imperfect
    subtype/direction. Set False for the the strict-only path.
    """
    cb = ContextBuilder(neighbor_index_path=DATA / f"neighbor_index_{split}.parquet")

    fam_before = Counter()
    fam_after = Counter()
    tier_counts = Counter()
    n_dropped = 0
    n_total = 0
    records = []

    for cp in critic_paths:
        print(f"[merge] reading {cp}")
        with cp.open() as f:
            for line in f:
                w = json.loads(line)
                n_total += 1
                pid = w["pair_id"]
                win = w["winner_candidate"]
                fam_before[win.get("gold_family") or "?"] += 1

                score = w["score"]["min_plus"]
                strict = bool(win.get("passed"))
                tier = win.get("tier", "drop")

                # Drop criteria:
                #   1. PRM score below threshold, OR
                #   2. tier == 'drop' (failed critical gates or unreasoned abstention), OR
                #   3. require_strict=True and tier != 'full_correct', OR
                #   4. keep_near_miss=False and tier not in {full_correct, abstention}
                # Note: abstention traces are always retained when keep_near_miss=True
                # because they carry distinct calibrated-uncertainty signal. They are
                # excluded from require_strict (strict = only confidently-correct).
                if score < min_score or tier == "drop":
                    n_dropped += 1
                    continue
                if require_strict and tier != "full_correct":
                    n_dropped += 1
                    continue
                if (not keep_near_miss) and tier not in ("full_correct", "abstention"):
                    n_dropped += 1
                    continue

                parsed = win.get("parsed")
                if parsed is None:
                    n_dropped += 1
                    continue

                ctx = cb.build(pid)
                msgs = build_prompt(ctx)
                assistant_json = json.dumps(parsed, separators=(",", ":"))

                records.append({
                    "pair_id": pid,
                    "family": win.get("gold_family"),
                    "subtype": win.get("gold_subtype"),
                    "direction_tag": win.get("gold_direction"),
                    "polarity": win.get("gold_polarity"),
                    "messages": [
                        {"role": "system", "content": msgs["system"]},
                        {"role": "user", "content": msgs["user"]},
                        {"role": "assistant", "content": assistant_json},
                    ],
                    "critic_score": score,
                    "qc_strict_passed": strict,
                    "tier": tier,
                    "sample_weight": TIER_WEIGHTS.get(tier, 0.25),
                    "n_candidates": w.get("n_candidates", 1),
                })
                fam_after[win.get("gold_family") or "?"] += 1
                tier_counts[tier] += 1

    # Dedup by pair_id (keep highest score)
    best_per_pair: dict[str, dict] = {}
    for r in records:
        cur = best_per_pair.get(r["pair_id"])
        if cur is None or r["critic_score"] > cur["critic_score"]:
            best_per_pair[r["pair_id"]] = r
    records = list(best_per_pair.values())
    print(f"[merge] {len(records):,} unique pairs survived "
          f"(dropped {n_dropped} / {n_total} raw)")

    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Audit report
    audit = ROOT / "outputs" / "audit" / "b5_merge_report.md"
    lines = [
        f"# B5 teacher_clean merge report",
        "",
        f"- Inputs:  {len(critic_paths)} critic file(s)",
        f"- Raw winners considered: {n_total:,}",
        f"- Dropped (score < {min_score} or strict-fail): {n_dropped:,}",
        f"- Final teacher_clean size: {len(records):,} pairs",
        f"- Output: `{out_path}`",
        "",
        "## Family balance — before vs after merge",
        "",
        "| Family | Before | After | Retention |",
        "|--------|--------|-------|-----------|",
    ]
    total_before = sum(fam_before.values())
    total_after = sum(fam_after.values())
    all_fams = sorted(set(fam_before) | set(fam_after))
    for fam in all_fams:
        b, a = fam_before[fam], fam_after[fam]
        ret = f"{100*a/b:.1f}%" if b else "—"
        lines.append(f"| {fam} | {b:,} | {a:,} | {ret} |")
    lines += ["", f"| TOTAL | {total_before:,} | {total_after:,} | "
                 f"{100*total_after/max(1,total_before):.1f}% |", ""]

    lines += ["", "## Tier breakdown (post-merge)", "",
              "| Tier | Count | Share | Sample weight |",
              "|------|-------|-------|---------------|"]
    n_r = max(1, len(records))
    for tier in ("full_correct", "family_correct", "abstention", "near_miss"):
        c = tier_counts.get(tier, 0)
        lines.append(f"| {tier} | {c:,} | {100*c/n_r:.1f}% | "
                     f"{TIER_WEIGHTS[tier]:.2f} |")
    lines.append("")
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("\n".join(lines))
    print(f"[merge] {audit.relative_to(ROOT)}")

    # Mirror-pair sidecar (P4 substrate).  Honor $DDI_OUTPUTS so it lands
    # on $SCRATCH on the cluster rather than the project filesystem.
    teacher_out = Path(os.environ.get("DDI_OUTPUTS", ROOT / "outputs")) / "teacher"
    teacher_out.mkdir(parents=True, exist_ok=True)
    mirror = teacher_out / "mirror_pairs.jsonl"
    by_key: dict[frozenset, list[str]] = defaultdict(list)
    for r in records:
        # pair_id is already canonical "DB_small|DB_big"; the mirror of A|B
        # shares the same set.  So mirrors = pairs with the same drug set
        # AND opposite direction_tag.
        a, b = r["pair_id"].split("|")
        by_key[frozenset([a, b])].append(r["pair_id"])
    n_with_mirror = 0
    with mirror.open("w") as f:
        for key, pids in by_key.items():
            if len(pids) >= 2:
                for pid in pids:
                    f.write(json.dumps({"pair_id": pid,
                                        "mirror_pair_ids": [q for q in pids if q != pid]}) + "\n")
                n_with_mirror += len(pids)
    print(f"[merge] mirror-pair sidecar: {n_with_mirror:,} records ")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--critic", nargs="+", required=True,
                   help="One or more critic_*.jsonl files")
    p.add_argument("--split", default="subset25k")
    p.add_argument("--output", required=True)
    p.add_argument("--min_score", type=float, default=0.60)
    p.add_argument("--no_strict", dest="require_strict", action="store_false",
                   help="Accept tiers below full_correct (family_correct, near_miss). "
                        "Default: strict (full_correct only). Recommended: pass this "
                        "flag with --keep_near_miss for the tier-weighted path.")
    p.add_argument("--no_near_miss", dest="keep_near_miss", action="store_false",
                   help="Exclude near_miss tier even with --no_strict")
    p.set_defaults(require_strict=True, keep_near_miss=True)
    args = p.parse_args()
    merge([Path(p) for p in args.critic], args.split, Path(args.output),
          args.min_score, args.require_strict, args.keep_near_miss)
