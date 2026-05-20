"""evaluation data -- Counterfactual PK-flip perturbations (CfS input set).

Motivation
----------
an earlier iteration of this project could not distinguish between mechanism-aware
reasoning and memorization.  A trustworthy DDI model should change
its prediction when a key PK flag flips -- e.g., if drug A is a
CYP3A4 inhibitor and we (counterfactually) tell it "A is NOT a CYP3A4
inhibitor", the model should reconsider.  Conversely, flipping an
*irrelevant* flag (say, P-gp substrate status when the interaction
goes through CYP3A4) should barely move predictions.

The CfS metric (`src/metrics/cfs.py`) measures this:

    cfs_gap = cfs_relevant - cfs_null

This builder emits the perturbation specifications that drive CfS
evaluation.  The inference pipeline (Phase D) consumes this file,
builds context bundles twice per record (original + perturbed),
predicts under both, and the eval harness computes CfS.

Output record format (JSONL)
----------------------------
    {
      "pair_id":             "DB00001|DB06605",
      "a_id":                "DB00001",
      "b_id":                "DB06605",
      "gold_family":         "PK_Metabolism",
      "gold_subtype":        "inhibits_CYP3A4",
      "perturbation_id":     "cyp3a4_inh_A_flip",
      "perturbation_type":   "cyp3a4_inh",   # matches column in pk_features
      "perturbed_drug":      "A" | "B",
      "perturbed_drugbank":  "DB00001",
      "original_value":      true,
      "perturbed_value":     false,
      "relevant":            true,             # key flag -> prediction must flip
      "split":               "test" | "val",
    }

One pair gets up to 2 records: one RELEVANT perturbation and one NULL
perturbation.  For each pair we emit both so the CfS metric has
matched relevant/null samples per pair (fair comparison).

Relevance heuristic (inferred from PK features, not subtype text)
-----------------------------------------------------------------
DrugBank's PK subtype labels are coarse (`metabolism`, `excretion_rate`,
`serum_concentration`, `absorption_change`, ...) -- they don't name the
specific enzyme or transporter.  We infer the candidate *mechanism flag*
from the PK feature table:

    For each PK pair (A, B) with family in {PK_Metabolism, ...}:
        For each enzyme / transporter E:
            if subtype is `metabolism`, `serum_concentration`, `excretion_rate`:
                RELEVANT iff   A has E_inh or E_ind True
                               AND B has E_sub True
                               (or the direction-reversed version)
            if subtype is `absorption_change` or `bioavailability`:
                prefer transporters (P-gp, BCRP) with the same rule
            if subtype is `protein_binding`:
                no PK-flag relevant -- we skip (protein binding isn't a
                boolean flag in our feature table).

A flag is NULL iff:
    - It's a legal PK flag True on the subject drug (so flipping has
      a real mechanistic meaning), AND
    - Its enzyme/transporter prefix is DIFFERENT from the relevant flag's
      (so the mechanism is plausibly independent), AND
    - The object drug is NOT a substrate of that enzyme/transporter
      (confirming no alternative interaction pathway).

CLI
---
    python -m src.data.build_counterfactual \
        --labels     data_processed/labels_hierarchical.parquet \
        --pk         data_processed/pk_features.parquet \
        --splits     data_processed/splits/subset25k.json \
        --split_section test \
        --output     data_processed/counterfactual_test.jsonl \
        --seed       42 \
        --max_per_pair 2

Output summary
--------------
On subset25k test (~5 k pairs), this typically produces ~6 k records
(most PK pairs have a natural relevant + null match; some pairs are
skipped for lack of either).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


# Enzyme / transporter prefixes we know about.  These names map to the
# prefix portion of the PK feature columns (e.g. "cyp3a4" -> cyp3a4_inh,
# cyp3a4_ind, cyp3a4_sub).
_ENZYMES: list[str] = [
    "cyp1a2", "cyp2a6", "cyp2b6", "cyp2c8", "cyp2c9", "cyp2c19",
    "cyp2d6", "cyp2e1", "cyp3a4", "cyp3a5",
]
_TRANSPORTERS: list[str] = [
    "p_gp", "bcrp", "oatp1b1", "oatp1b3", "oat1", "oat3", "oct1", "oct2",
]
_ALL_PREFIXES: list[str] = _ENZYMES + _TRANSPORTERS

# Which subtypes allow which prefixes as candidate mechanisms.
_SUBTYPE_PREFIX_CANDIDATES: dict[str, list[str]] = {
    # CYP-metabolism effects: enzymes only
    "metabolism":                     _ENZYMES,
    "serum_concentration":            _ENZYMES + _TRANSPORTERS,
    "excretion_rate":                 _TRANSPORTERS + _ENZYMES,
    "active_metabolite_serum_conc":   _ENZYMES,
    # Absorption/bioavailability: transporters first, CYP3A4 in gut as fallback
    "absorption_change":              ["p_gp", "bcrp", "cyp3a4"],
    "bioavailability":                ["p_gp", "bcrp", "cyp3a4"],
    # Protein binding isn't a boolean PK flag -> no CF perturbation
    "protein_binding":                [],
}

_PK_FAMILIES = {"PK_Metabolism", "PK_Absorption", "PK_Excretion", "PK_Distribution"}
_ALL_PK_FLAG_COLS: list[str] = sorted({
    f"{p}_{k}" for p in _ALL_PREFIXES for k in ("inh", "ind", "sub")
})


def _load_pk(pk_path: Path) -> dict[str, dict[str, bool]]:
    """Load PK features -> dict drugbank_id -> {flag_col: bool}."""
    tbl = pq.read_table(pk_path, columns=["drugbank_id"] + _ALL_PK_FLAG_COLS).to_pylist()
    out: dict[str, dict[str, bool]] = {}
    for row in tbl:
        out[row["drugbank_id"]] = {c: bool(row.get(c, False)) for c in _ALL_PK_FLAG_COLS}
    return out


def _load_splits(splits_path: Path | None, section: str) -> set[str] | None:
    """Load a parquet split manifest.  Columns: pair_id, split, family.
    Accepts both paths to a specific manifest file AND the old JSON shape
    (for forward-compat with fixture tests).
    """
    if not splits_path or not splits_path.exists():
        return None
    if splits_path.suffix == ".parquet":
        tbl = pq.read_table(splits_path, columns=["pair_id", "split"]).to_pylist()
        return {r["pair_id"] for r in tbl if r.get("split") == section}
    with open(splits_path) as f:
        manifest = json.load(f)
    if section not in manifest:
        return None
    return set(manifest[section])


def _infer_relevant_mechanism(
    subtype: str,
    a_flags: dict[str, bool],
    b_flags: dict[str, bool],
    declared_subject: str | None,
    a_id: str,
    b_id: str,
) -> tuple[str, str, str, str] | None:
    """Infer (flag_col, prefix, subject_side, subject_drug) for a pair.

    Scans candidate enzyme/transporter prefixes (subtype-specific order)
    and picks the FIRST where the implied mechanism is consistent:

        <subject>_<prefix>_<inh|ind> == True  AND  <object>_<prefix>_sub == True

    If `declared_subject` is set, only that side is considered as subject.
    Otherwise we try both orderings.
    """
    candidates = _SUBTYPE_PREFIX_CANDIDATES.get(subtype, _ALL_PREFIXES)
    if not candidates:
        return None

    # Candidate subject sides:
    sides: list[tuple[str, str, dict, dict]] = []
    if declared_subject == a_id:
        sides = [("A", a_id, a_flags, b_flags)]
    elif declared_subject == b_id:
        sides = [("B", b_id, b_flags, a_flags)]
    else:
        sides = [
            ("A", a_id, a_flags, b_flags),
            ("B", b_id, b_flags, a_flags),
        ]

    for prefix in candidates:
        inh_col, ind_col, sub_col = f"{prefix}_inh", f"{prefix}_ind", f"{prefix}_sub"
        for side, drug, subj_flags, obj_flags in sides:
            if not obj_flags.get(sub_col, False):
                continue
            if subj_flags.get(inh_col, False):
                return inh_col, prefix, side, drug
            if subj_flags.get(ind_col, False):
                return ind_col, prefix, side, drug
    return None


def _pick_null_flag(subj_flags: dict[str, bool], obj_flags: dict[str, bool],
                    relevant_prefix: str, rng: random.Random) -> str | None:
    """Pick a flag that is:
      - True on subject,
      - from a DIFFERENT prefix than `relevant_prefix`,
      - and the object is NOT a substrate of it (so there's no alternate
        mechanism accidentally introduced by our perturbation).
    """
    candidates: list[str] = []
    for prefix in _ALL_PREFIXES:
        if prefix == relevant_prefix:
            continue
        if obj_flags.get(f"{prefix}_sub", False):
            continue
        for k in ("inh", "ind"):
            col = f"{prefix}_{k}"
            if subj_flags.get(col, False):
                candidates.append(col)
    if not candidates:
        return None
    return rng.choice(candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pk",     required=True)
    ap.add_argument("--splits", default=None,
                    help="Optional splits manifest (data_processed/splits/<split>.json).")
    ap.add_argument("--split_section", default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--max_per_pair", type=int, default=2,
                    help="Up to N perturbations per pair (1 relevant + 1 null).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pk = _load_pk(Path(args.pk))
    print(f"[cf] PK features for {len(pk):,} drugs")

    split_pairs = _load_splits(Path(args.splits), args.split_section) if args.splits else None
    if split_pairs is not None:
        print(f"[cf] split filter: {len(split_pairs):,} pair_ids")

    tbl = pq.read_table(args.labels, columns=[
        "pair_id", "a_id", "b_id", "family", "subtype", "bidirectional",
        "subject_drugbank_id",
    ]).to_pylist()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_in = n_rel = n_null = n_out = 0
    skip_reasons: Counter = Counter()
    by_perturbation: Counter = Counter()

    with open(args.output, "w") as f:
        for row in tbl:
            n_in += 1
            pair_id = row["pair_id"]
            if split_pairs is not None and pair_id not in split_pairs:
                continue
            if row.get("family") not in _PK_FAMILIES:
                skip_reasons["non_pk_family"] += 1
                continue
            subtype = row.get("subtype") or ""

            a_id, b_id = row["a_id"], row["b_id"]
            a_flags = pk.get(a_id, {})
            b_flags = pk.get(b_id, {})
            if not a_flags or not b_flags:
                skip_reasons["pk_missing"] += 1
                continue

            declared_subject = row.get("subject_drugbank_id")
            # For bidirectional / no-declared cases we let the inference pick either side.
            inferred = _infer_relevant_mechanism(
                subtype, a_flags, b_flags, declared_subject, a_id, b_id,
            )
            if inferred is None:
                skip_reasons[f"no_mechanism_inferred:{subtype}"] += 1
                continue
            rel_flag, rel_prefix, subj_side, subj_drug = inferred

            subj_flags = a_flags if subj_side == "A" else b_flags
            obj_flags  = b_flags if subj_side == "A" else a_flags

            rel_rec = {
                "pair_id":            pair_id,
                "a_id":               a_id,
                "b_id":               b_id,
                "gold_family":        row["family"],
                "gold_subtype":       subtype,
                "perturbation_id":    f"{rel_flag}_{subj_side}_flip",
                "perturbation_type":  rel_flag,
                "perturbation_prefix": rel_prefix,
                "perturbed_drug":     subj_side,
                "perturbed_drugbank": subj_drug,
                "original_value":     True,
                "perturbed_value":    False,
                "relevant":           True,
                "split":              args.split_section,
            }
            f.write(json.dumps(rel_rec) + "\n")
            n_out += 1; n_rel += 1
            by_perturbation[rel_flag] += 1

            if args.max_per_pair >= 2:
                null_flag = _pick_null_flag(subj_flags, obj_flags, rel_prefix, rng)
                if null_flag:
                    null_prefix = null_flag.rsplit("_", 1)[0]
                    null_rec = {
                        "pair_id":            pair_id,
                        "a_id":               a_id,
                        "b_id":               b_id,
                        "gold_family":        row["family"],
                        "gold_subtype":       subtype,
                        "perturbation_id":    f"{null_flag}_{subj_side}_flip",
                        "perturbation_type":  null_flag,
                        "perturbation_prefix": null_prefix,
                        "perturbed_drug":     subj_side,
                        "perturbed_drugbank": subj_drug,
                        "original_value":     True,
                        "perturbed_value":    False,
                        "relevant":           False,
                        "split":              args.split_section,
                    }
                    f.write(json.dumps(null_rec) + "\n")
                    n_out += 1; n_null += 1
                    by_perturbation[null_flag] += 1
                else:
                    skip_reasons["no_null_candidate"] += 1

    print(f"[cf] input rows: {n_in:,}")
    print(f"[cf] emitted   : {n_out:,}  (relevant={n_rel:,}, null={n_null:,})")
    print(f"[cf] skip reasons: {dict(skip_reasons)}")
    print(f"[cf] top perturbations: {by_perturbation.most_common(10)}")

    if args.report:
        rep = Path(args.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        with open(rep, "w") as f:
            f.write("# Counterfactual PK-flip dataset (CfS input)\n\n")
            f.write(f"- Source labels: `{args.labels}`\n")
            f.write(f"- Split: `{args.split_section}` ({args.splits or 'all'})\n")
            f.write(f"- Relevant perturbations: **{n_rel:,}**\n")
            f.write(f"- Null perturbations:     **{n_null:,}**\n")
            f.write(f"- Total records:          **{n_out:,}**\n\n")
            f.write("## Skip reasons\n\n| reason | n |\n|---|---:|\n")
            for k, v in sorted(skip_reasons.items()):
                f.write(f"| `{k}` | {v:,} |\n")
            f.write("\n## Top perturbation flags\n\n| flag | n |\n|---|---:|\n")
            for flag, n in by_perturbation.most_common():
                f.write(f"| `{flag}` | {n:,} |\n")
        print(f"[cf] wrote report {rep}")


if __name__ == "__main__":
    main()
