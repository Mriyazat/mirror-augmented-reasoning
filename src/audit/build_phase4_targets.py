"""Build targeted Phase 4.2 regeneration manifest from Phase 4.1 audit.

The output is deliberately small and explicit:
  - manifest JSONL: one pair_id per row with audit cluster metadata.
  - guidance JSONL: per-pair teacher prompt addendum.
  - integrity JSON: source hashes/counts to prevent stale-file mistakes.

This is NOT broad self-distillation. It targets the observed failure clusters
where the Phase 3 student flips mechanistic rare-family labels into
AdverseRisk or makes direction/subtype-only mistakes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CLUSTERS = (
    "pk_metabolism_mislabeled_as_adverse",
    "pk_excretion_mislabeled_as_adverse",
    "efficacy_mislabeled_as_adverse",
    "pk_distribution_mislabeled_as_adverse",
    "pd_activity_mislabeled_as_adverse",
    "axis_ambiguity_same_subtype",
    "direction_only_error",
    "subtype_only_error",
)


GUIDANCE_BY_CLUSTER = {
    "pk_metabolism_mislabeled_as_adverse": """\
This pair is being regenerated because the student confused a CYP/enzyme-mediated
PK_Metabolism interaction with AdverseRisk. Decide the family by MECHANISM, not
clinical consequence: if one drug inhibits/induces/is a substrate of CYP/UGT or
another metabolic enzyme that changes the other drug's metabolism, choose
PK_Metabolism/metabolism even if the downstream consequence is higher exposure
or adverse effects. Use AdverseRisk only when the primary evidence supports a
clinical adverse event mechanism rather than a metabolic mechanism.""",
    "pk_excretion_mislabeled_as_adverse": """\
This pair is being regenerated because the student confused a renal/biliary
clearance interaction with AdverseRisk. Decide the family by MECHANISM: if the
evidence involves renal excretion, OAT/OCT/OATP/BCRP transport, biliary
clearance, or reduced/increased excretion rate, choose PK_Excretion/excretion_rate
even if exposure changes can raise adverse risk. Use AdverseRisk only when the
primary evidence is a clinical adverse event mechanism.""",
    "pk_distribution_mislabeled_as_adverse": """\
This pair is being regenerated because the student confused a distribution or
protein-binding interaction with AdverseRisk. Decide the family by MECHANISM:
if the evidence involves albumin, alpha-1-acid glycoprotein, serum/plasma
concentration, protein binding, carrier binding, or displacement, choose
PK_Distribution with the appropriate subtype. Do not relabel it AdverseRisk
just because serum concentration changes can cause toxicity.""",
    "pd_activity_mislabeled_as_adverse": """\
This pair is being regenerated because the student confused a pharmacodynamic
activity interaction with AdverseRisk. Decide the family by MECHANISM: if both
drugs act on related receptors/pathways and the interaction is additive,
synergistic, or antagonistic WITHOUT a PK shift, choose PD_Activity with the
matching subtype. Reserve AdverseRisk for cases whose primary label is the
clinical adverse event rather than the pharmacodynamic mechanism.""",
    "efficacy_mislabeled_as_adverse": """\
This pair is being regenerated because the student confused altered therapeutic
efficacy with AdverseRisk. Decide the family by OUTCOME AXIS: if the evidence
supports increased/decreased therapeutic efficacy or diagnostic effectiveness
without a primary adverse-event mechanism, choose Efficacy. Do not collapse
efficacy changes into AdverseRisk unless the evidence explicitly supports a
clinical adverse risk label.""",
    "axis_ambiguity_same_subtype": """\
This pair is being regenerated because the subtype/clinical phenomenon is
similar but the family axis was confused. Follow the the family definitions
strictly: choose PK_* for pharmacokinetic mechanisms, PD_Activity for additive
or antagonistic pharmacodynamic activity, Efficacy for therapeutic efficacy,
and AdverseRisk only when the primary label is increased adverse-event risk.
Mention the mechanism that justifies the family axis in the conclusion.""",
    "direction_only_error": """\
This pair is being regenerated because the student got the family/subtype but
flipped direction. Before the conclusion, add a `direction` step that explicitly
states which drug acts on which. Use `a_to_b` only when A changes B's behavior,
`b_to_a` only when B changes A's behavior, and `bidirectional` only for truly
symmetric mechanisms. The final_answer.direction_tag must match that step.""",
    "subtype_only_error": """\
This pair is being regenerated because the student got the family but selected
the wrong subtype. Use the subtype whitelist exactly. Pick the subtype whose
mechanism and clinical wording best match the evidence and neighbor labels;
avoid generic `adverse_effects` unless the evidence does not support a more
specific subtype.""",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_audit_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _parse_clusters(value: str | None) -> set[str]:
    if not value:
        return set(DEFAULT_CLUSTERS)
    return {x.strip() for x in value.split(",") if x.strip()}


def _append_gold_hint(base: str, row: dict[str, Any]) -> str:
    """Teacher generation normally avoids gold labels. Here the gold label is
    exactly the target for hard-negative correction, so we expose it explicitly
    as a supervised disambiguation hint. This is Phase 4 targeted repair, not
    blind teacher-as-reasoner generation.
    """
    return (
        base.strip()
        + "\n\nFor this targeted repair example, the intended target label is: "
        + f"family={row['gold_family']}, subtype={row['gold_subtype']}, "
        + f"direction_tag={row['gold_direction']}. Produce evidence-grounded "
        + "reasoning that justifies this label; if the evidence cannot justify "
        + "it, abstain rather than inventing support."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--clusters", default=None,
                   help="Comma-separated cluster names. Default: core Phase 4.2 clusters.")
    p.add_argument("--max_per_cluster", type=int, default=-1)
    p.add_argument("--include_gold_hint", action="store_true",
                   help="Append intended target label to guidance. Recommended for hard-negative repair.")
    p.add_argument("--split_name", default="phase4_targeted")
    args = p.parse_args()

    audit_csv = Path(args.audit_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_audit_csv(audit_csv)
    wanted = _parse_clusters(args.clusters)

    selected: list[dict[str, Any]] = []
    counts: Counter = Counter()
    seen: set[str] = set()
    for row in rows:
        cluster = row.get("cluster")
        pid = row.get("pair_id")
        if cluster not in wanted or not pid or pid in seen:
            continue
        if args.max_per_cluster > 0 and counts[cluster] >= args.max_per_cluster:
            continue
        selected.append(row)
        counts[cluster] += 1
        seen.add(pid)

    manifest_path = out_dir / f"manifest_{args.split_name}.jsonl"
    guidance_path = out_dir / f"guidance_{args.split_name}.jsonl"
    integrity_path = out_dir / f"integrity_{args.split_name}.json"

    with manifest_path.open("w") as f:
        for row in selected:
            f.write(json.dumps({
                "pair_id": row["pair_id"],
                "cluster": row["cluster"],
                "gold_family": row["gold_family"],
                "gold_subtype": row["gold_subtype"],
                "gold_direction": row["gold_direction"],
                "pred_family": row["pred_family"],
                "pred_subtype": row["pred_subtype"],
                "pred_direction": row["pred_direction"],
            }) + "\n")

    with guidance_path.open("w") as f:
        for row in selected:
            guidance = GUIDANCE_BY_CLUSTER.get(row["cluster"], "").strip()
            if args.include_gold_hint:
                guidance = _append_gold_hint(guidance, row)
            f.write(json.dumps({
                "pair_id": row["pair_id"],
                "cluster": row["cluster"],
                "guidance": guidance,
            }) + "\n")

    integrity = {
        "audit_csv": str(audit_csv),
        "audit_csv_sha256": _sha256(audit_csv),
        "audit_rows": len(rows),
        "selected_rows": len(selected),
        "selected_unique_pair_ids": len(seen),
        "clusters": dict(counts.most_common()),
        "manifest": str(manifest_path),
        "guidance": str(guidance_path),
        "include_gold_hint": bool(args.include_gold_hint),
        "max_per_cluster": args.max_per_cluster,
    }
    integrity_path.write_text(json.dumps(integrity, indent=2) + "\n")

    print(f"[targets] wrote {manifest_path}")
    print(f"[targets] wrote {guidance_path}")
    print(f"[targets] wrote {integrity_path}")
    print(f"[targets] selected {len(selected)} pairs")
    for cluster, count in counts.most_common():
        print(f"[targets]   {cluster:45s} {count:5d}")


if __name__ == "__main__":
    main()
