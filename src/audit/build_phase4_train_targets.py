"""Build non-leaky Phase 4 training-side repair targets.

The first Phase 4 targeted run proved the repair mechanism on Phase-3
validation errors. That corpus is useful for diagnosis but cannot be used for
training because it overlaps the evaluation set. This script applies the same
repair idea to Phase-3 *training* records only:

  - reads the commits-only Phase-C train corpus;
  - excludes every pair_id present in the Phase-C val corpus;
  - prioritizes rare/ambiguous families and low-consensus records;
  - emits a manifest + per-pair guidance JSONL for teacher regeneration.

The guidance intentionally includes the intended V4 label because this is a
supervised hard-repair corpus, not blind synthetic self-distillation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGET_FAMILIES = (
    "PK_Metabolism",
    "PK_Excretion",
    "Efficacy",
    "PD_Activity",
    "PK_Distribution",
    "PK_Absorption",
)

DEFAULT_QUOTAS = {
    # The observed Phase-3 failure was mostly rare-family -> AdverseRisk.
    # Quotas are deliberately heavier on mechanism families that had low recall.
    "PK_Metabolism": 900,
    "PK_Excretion": 500,
    "Efficacy": 400,
    "PD_Activity": 400,
    "PK_Distribution": 300,
    "PK_Absorption": 150,
    # Keep a small AdverseRisk anchor set so the repair corpus does not teach
    # the student to under-predict AdverseRisk after focal/class-balanced SFT.
    "AdverseRisk": 250,
}

GUIDANCE_BY_FAMILY = {
    "PK_Metabolism": """\
This is a training-side targeted repair example for PK_Metabolism. Decide the
family by MECHANISM, not by downstream clinical consequence. If one drug
inhibits, induces, or is a substrate of CYP/UGT/metabolic enzymes that changes
the other drug's metabolism, choose PK_Metabolism/metabolism. Do not collapse
the answer into AdverseRisk just because altered exposure can raise toxicity.""",
    "PK_Excretion": """\
This is a training-side targeted repair example for PK_Excretion. Decide the
family by MECHANISM. If the evidence involves renal/biliary clearance,
OAT/OCT/OATP/BCRP transport, or excretion-rate change, choose
PK_Excretion/excretion_rate even when the downstream effect is higher exposure
or clinical risk.""",
    "PK_Distribution": """\
This is a training-side targeted repair example for PK_Distribution. If the
evidence involves albumin, alpha-1-acid glycoprotein, protein binding,
carrier binding, serum/plasma concentration, or displacement, choose
PK_Distribution with the appropriate subtype. Do not relabel it AdverseRisk
only because distribution changes can cause adverse effects.""",
    "PD_Activity": """\
This is a training-side targeted repair example for PD_Activity. If both drugs
act through related receptors/pathways and the interaction is additive,
synergistic, or antagonistic without a PK shift, choose PD_Activity with the
matching subtype. Reserve AdverseRisk for primary clinical adverse-event labels.""",
    "Efficacy": """\
This is a training-side targeted repair example for Efficacy. If the evidence
supports altered therapeutic efficacy or diagnostic effectiveness without a
primary adverse-event mechanism, choose Efficacy. Do not collapse therapeutic
benefit/loss into AdverseRisk unless adverse risk is the primary label.""",
    "PK_Absorption": """\
This is a training-side targeted repair example for PK_Absorption. If the
evidence involves gastrointestinal absorption, bioavailability, intestinal
transport, or P-gp-mediated absorption, choose PK_Absorption with the matching
subtype instead of AdverseRisk.""",
    "AdverseRisk": """\
This is an AdverseRisk anchor example. Choose AdverseRisk only when the primary
evidence supports increased clinical adverse-event risk, and name the adverse
event subtype precisely. Do not overuse AdverseRisk for interactions whose
primary mechanism is PK_Metabolism, PK_Excretion, PK_Distribution, PD_Activity,
or Efficacy.""",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_quotas(value: str | None) -> dict[str, int]:
    quotas = dict(DEFAULT_QUOTAS)
    if not value:
        return quotas
    out: dict[str, int] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(
                f"Bad quota component {part!r}; use Family=N,Family=N"
            )
        fam, n = part.split("=", 1)
        out[fam.strip()] = int(n)
    return out


def _record_priority(rec: dict[str, Any]) -> tuple:
    """Higher-priority records sort first.

    We want examples that most directly counter Phase-3 failures:
      - lower k_family (less consensus in original corpus),
      - family_correct tier (family right but subtype/direction imperfect),
      - higher existing quality/weight within those bands.
    """
    consensus = rec.get("consensus") or {}
    k_family = int(consensus.get("k_family") or 1)
    tier = rec.get("tier") or ""
    quality = float(rec.get("quality_score") or 0.0)
    weight = float(rec.get("sample_weight") or 0.0)
    tier_rank = {"family_correct": 2, "full_correct": 1}.get(tier, 0)
    return (tier_rank, -k_family, quality, weight, rec.get("pair_id") or "")


def _guidance_for(rec: dict[str, Any]) -> str:
    fam = rec.get("family")
    base = GUIDANCE_BY_FAMILY.get(fam, "").strip()
    return (
        base
        + "\n\nFor this supervised repair example, the intended V4 label is: "
        + f"family={rec.get('family')}, subtype={rec.get('subtype')}, "
        + f"direction_tag={rec.get('direction_tag')}. Produce evidence-grounded "
        + "reasoning that justifies this label; if the evidence cannot justify "
        + "it, abstain rather than inventing support."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split_name", default="phase4_train_targeted")
    p.add_argument(
        "--quotas",
        default=None,
        help=(
            "Comma-separated Family=N quotas. Default is tuned for Phase-3 "
            "rare-family -> AdverseRisk failures."
        ),
    )
    args = p.parse_args()

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quotas = _parse_quotas(args.quotas)

    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    val_ids = {r.get("pair_id") for r in val_rows}

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_val_overlap = 0
    skipped_tier = 0
    seen: set[str] = set()
    for rec in train_rows:
        pid = rec.get("pair_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        if pid in val_ids:
            excluded_val_overlap += 1
            continue
        if rec.get("tier") not in {"full_correct", "family_correct"}:
            skipped_tier += 1
            continue
        fam = rec.get("family")
        if fam in quotas:
            by_family[fam].append(rec)

    selected: list[dict[str, Any]] = []
    selected_counts: Counter = Counter()
    for fam, quota in quotas.items():
        candidates = sorted(
            by_family.get(fam, []),
            key=_record_priority,
            reverse=True,
        )
        take = candidates[:quota]
        selected.extend(take)
        selected_counts[fam] = len(take)

    # Stable order: hardest/rarest signal first but deterministic.
    selected = sorted(
        selected,
        key=lambda r: (
            -selected_counts[r.get("family")],
            r.get("family") or "",
            r.get("pair_id") or "",
        ),
    )

    manifest_path = out_dir / f"manifest_{args.split_name}.jsonl"
    guidance_path = out_dir / f"guidance_{args.split_name}.jsonl"
    integrity_path = out_dir / f"integrity_{args.split_name}.json"

    with manifest_path.open("w") as f:
        for rec in selected:
            f.write(json.dumps({
                "pair_id": rec["pair_id"],
                "source": "phase_c_train",
                "family": rec.get("family"),
                "subtype": rec.get("subtype"),
                "direction_tag": rec.get("direction_tag"),
                "tier": rec.get("tier"),
                "sample_weight": rec.get("sample_weight"),
                "quality_score": rec.get("quality_score"),
                "k_family": (rec.get("consensus") or {}).get("k_family"),
            }) + "\n")

    with guidance_path.open("w") as f:
        for rec in selected:
            f.write(json.dumps({
                "pair_id": rec["pair_id"],
                "source": "phase_c_train",
                "family": rec.get("family"),
                "guidance": _guidance_for(rec),
            }) + "\n")

    integrity = {
        "train_file": str(train_path),
        "train_sha256": _sha256(train_path),
        "train_rows": len(train_rows),
        "train_unique_pair_ids": len(seen),
        "val_file": str(val_path),
        "val_sha256": _sha256(val_path),
        "val_rows": len(val_rows),
        "val_unique_pair_ids": len(val_ids),
        "excluded_val_overlap": excluded_val_overlap,
        "skipped_tier": skipped_tier,
        "quotas": quotas,
        "selected_rows": len(selected),
        "selected_unique_pair_ids": len({r["pair_id"] for r in selected}),
        "selected_by_family": dict(selected_counts),
        "manifest": str(manifest_path),
        "guidance": str(guidance_path),
        "split_name": args.split_name,
    }
    integrity_path.write_text(json.dumps(integrity, indent=2) + "\n")

    print(f"[train_targets] wrote {manifest_path}")
    print(f"[train_targets] wrote {guidance_path}")
    print(f"[train_targets] wrote {integrity_path}")
    print(f"[train_targets] selected {len(selected)} train-side pairs")
    print(f"[train_targets] excluded_val_overlap={excluded_val_overlap}")
    for fam, count in selected_counts.most_common():
        print(f"[train_targets]   {fam:18s} {count:5d} / quota {quotas[fam]}")


if __name__ == "__main__":
    main()
