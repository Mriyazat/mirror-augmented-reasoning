"""evaluation data -- Adversarial RIS dataset (retrieval contradiction).

Motivation
----------
An earlier model did NOT separate retrieval-use from memorized priors: giving
it a wrong retrieval bundle hardly changed its predictions, and giving
it a right retrieval bundle barely improved them.  The RIS metric
(`src/metrics/ris.py`) is designed to detect this:

    RIS = accuracy(TRUE evidence) - accuracy(ADVERSARIAL evidence)

A model that genuinely uses retrieval will score HIGH on RIS: it
benefits from true evidence AND is misled by adversarial (contradictory)
evidence.  A model that ignores retrieval will score LOW (both accuracies
collapse to its prior).  A model that OVER-trusts retrieval (flips
whenever context changes) will score HIGH even when its prior was wrong
-- so we also report Delta_adv = acc(no-ev) - acc(adv-ev) as a sanity
check.

Output record format (JSONL)
----------------------------
    {
      "pair_id":              "DB00001|DB06605",
      "a_id":                 "DB00001",
      "b_id":                 "DB06605",
      "gold_family":          "PK_Metabolism",
      "gold_subtype":         "metabolism",
      "true_context_ids":     [ ... evidence ids supporting gold ...],
      "adv_context_ids":      [ ... evidence ids supporting a wrong family ...],
      "adv_target_family":    "PD_Receptor",          # what the adv set points to
      "adv_strategy":         "cross_family_swap" | "enzyme_swap" | "null_ctx",
      "split":                "test"
    }

For each pair we emit exactly ONE adversarial record (the single
strongest contradiction) to keep the RIS estimator clean -- mixing
multiple strategies per pair biases the corpus-level RIS toward
whichever strategy produced more pairs.

Adversarial strategies
----------------------
1.  **cross_family_swap** (primary)
    Build a retrieval bundle that is COHERENT but points to a plausibly
    wrong family.  E.g. for a PK_Metabolism pair, inject evidence that
    emphasizes PD_Receptor overlap (shared downstream receptor protein,
    ATC-class similarity from drug_proteins / pair_signatures).  The
    idea: the adversary is an "expert liar" -- internally consistent
    but mechanistically wrong.

2.  **enzyme_swap** (PK-specific)
    For PK pairs, replace the inferred relevant enzyme (e.g. cyp3a4)
    with an irrelevant one (e.g. cyp2d6) that the pair does NOT share.
    This should confuse a model that blindly trusts retrieved PK flags.

3.  **null_ctx** (diagnostic -- empty context)
    Emit an empty evidence pool.  This is not strictly adversarial but
    feeds the RIS baseline decomposition (acc(no-ev)) that the eval
    harness uses.  We tag it separately and the eval harness pulls it
    into the RIS.prediction_no_ev slot.

True evidence reconstruction
----------------------------
We reconstruct a TRUE evidence bundle by mirroring the Phase B teacher
ContextBuilder logic (without actually importing it -- that would
pull in the retrieval vectorizer).  Specifically:
  - for PK pairs: cite shared enzyme_<prefix> + pk_flag_<prefix>_<role>
                  plus pair_signatures.pathway_shared
  - for PD pairs: cite shared protein_<ID> + overlapping pathway_<ID>
  - for Risk pairs: cite ATC-class overlap + shared adverse-reaction ids

The ids use the SAME naming as Phase B's context_ids so the eval
harness's evidence_resolution module resolves them uniformly.

CLI
---
    python -m src.data.build_adversarial \
        --labels        data_processed/labels_hierarchical.parquet \
        --pk            data_processed/pk_features.parquet \
        --signatures    data_processed/pair_signatures.parquet \
        --proteins      data_processed/drug_proteins.parquet \
        --pathways      data_processed/drug_pathways.parquet \
        --splits        data_processed/splits/manifest_pair_cold.parquet \
        --split_section test \
        --output        data_processed/adversarial_pair_cold_test.jsonl \
        --report        outputs/audit/adversarial_pair_cold_test.md
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


_PK_FAMILIES = {"PK_Metabolism", "PK_Absorption", "PK_Excretion", "PK_Distribution"}
_PD_FAMILIES = {"PD_Activity", "Efficacy"}
_RISK_FAMILIES = {"AdverseRisk"}

_CYP_PREFIXES = ["cyp1a2", "cyp2a6", "cyp2b6", "cyp2c8", "cyp2c9",
                 "cyp2c19", "cyp2d6", "cyp2e1", "cyp3a4", "cyp3a5"]
_TRANSPORTER_PREFIXES = ["p_gp", "bcrp", "oatp1b1", "oatp1b3", "oat1", "oat3", "oct1", "oct2"]
_ALL_PREFIXES = _CYP_PREFIXES + _TRANSPORTER_PREFIXES


def _load_pk(pk_path: Path) -> dict[str, dict[str, bool]]:
    cols = ["drugbank_id"] + [
        f"{p}_{k}" for p in _ALL_PREFIXES for k in ("inh", "ind", "sub")
    ]
    tbl = pq.read_table(pk_path, columns=cols).to_pylist()
    return {r["drugbank_id"]: {c: bool(r.get(c, False)) for c in cols[1:]}
            for r in tbl}


def _load_drug_proteins(path: Path) -> dict[str, list[str]]:
    """Map drugbank_id -> list of protein_id."""
    tbl = pq.read_table(path, columns=["drugbank_id", "protein_id"]).to_pylist()
    out: dict[str, list[str]] = defaultdict(list)
    for r in tbl:
        if r.get("protein_id"):
            out[r["drugbank_id"]].append(r["protein_id"])
    return out


def _load_drug_pathways(path: Path | None) -> dict[str, list[str]]:
    """drugbank_id -> list of pathway ids.  Uses SMPDB id as the canonical
    pathway identifier (matches the naming used elsewhere in the project)."""
    if path is None or not path.exists():
        return {}
    tbl = pq.read_table(path, columns=["drugbank_id", "smpdb_id"]).to_pylist()
    out: dict[str, list[str]] = defaultdict(list)
    for r in tbl:
        if r.get("smpdb_id"):
            out[r["drugbank_id"]].append(r["smpdb_id"])
    return out


def _load_signatures(path: Path) -> dict[str, dict]:
    tbl = pq.read_table(path).to_pylist()
    return {r["pair_id"]: r for r in tbl}


def _load_split_pairs(splits_path: Path | None, section: str) -> set[str] | None:
    if not splits_path or not splits_path.exists():
        return None
    tbl = pq.read_table(splits_path, columns=["pair_id", "split"]).to_pylist()
    return {r["pair_id"] for r in tbl if r.get("split") == section}


# ======================================================================
# TRUE-evidence builders (mirror Phase B ContextBuilder naming)
# ======================================================================
def _pk_true_ids(pair_id: str, a_id: str, b_id: str,
                 a_flags: dict[str, bool], b_flags: dict[str, bool]) -> tuple[list[str], str | None]:
    """Return (evidence_ids, inferred_prefix)."""
    ids: list[str] = []
    best_prefix: str | None = None
    for prefix in _ALL_PREFIXES:
        sub_a, sub_b = f"{prefix}_sub", f"{prefix}_sub"
        inh_a, inh_b = f"{prefix}_inh", f"{prefix}_inh"
        ind_a, ind_b = f"{prefix}_ind", f"{prefix}_ind"
        if b_flags.get(sub_b) and (a_flags.get(inh_a) or a_flags.get(ind_a)):
            if best_prefix is None:
                best_prefix = prefix
            ids.append(f"enzyme_{prefix.upper()}")
            if a_flags.get(inh_a):
                ids.append(f"pk_flag_{prefix}_inhibitor_A")
            if a_flags.get(ind_a):
                ids.append(f"pk_flag_{prefix}_inducer_A")
            ids.append(f"pk_flag_{prefix}_substrate_B")
        elif a_flags.get(sub_a) and (b_flags.get(inh_b) or b_flags.get(ind_b)):
            if best_prefix is None:
                best_prefix = prefix
            ids.append(f"enzyme_{prefix.upper()}")
            if b_flags.get(inh_b):
                ids.append(f"pk_flag_{prefix}_inhibitor_B")
            if b_flags.get(ind_b):
                ids.append(f"pk_flag_{prefix}_inducer_B")
            ids.append(f"pk_flag_{prefix}_substrate_A")
    return ids, best_prefix


def _pd_true_ids(a_id: str, b_id: str,
                 a_proteins: list[str], b_proteins: list[str],
                 a_pathways: list[str], b_pathways: list[str]) -> list[str]:
    shared_proteins = set(a_proteins) & set(b_proteins)
    shared_pathways = set(a_pathways) & set(b_pathways)
    out: list[str] = []
    for p in sorted(shared_proteins):
        out.append(f"protein_{p}")
    for pw in sorted(shared_pathways):
        out.append(f"pathway_{pw}")
    return out


# ======================================================================
# Adversarial constructors
# ======================================================================
def _adv_cross_family_swap(
    pair_id: str, a_id: str, b_id: str, gold_family: str,
    a_proteins: list[str], b_proteins: list[str],
    a_pathways: list[str], b_pathways: list[str],
    rng: random.Random,
) -> tuple[list[str], str] | None:
    """Build an adversarial bundle by swapping to a cross-family mechanism.

    For PK gold -> emphasize protein/pathway overlap (fake PD cues).
    For PD gold -> emphasize PK cues (fake pk_flag tokens pointing at a
    random enzyme).  Risk gold -> claim shared enzyme overlap.
    """
    if gold_family in _PK_FAMILIES:
        # Fake a PD-style context: cite one of each drug's own protein
        # targets as if they were "shared".  If either drug has no
        # protein entries, fall back to a synthesized pair.
        if not a_proteins or not b_proteins:
            return None
        fake_shared_protein = rng.choice(a_proteins)
        ids = [
            f"protein_{fake_shared_protein}",
            f"protein_{rng.choice(b_proteins) if b_proteins else fake_shared_protein}",
        ]
        if a_pathways and b_pathways:
            ids.append(f"pathway_{rng.choice(a_pathways)}")
            ids.append(f"pathway_{rng.choice(b_pathways)}")
        return ids, "PD_Receptor"
    elif gold_family in _PD_FAMILIES:
        # Fake a PK-style context: pick a random CYP/transporter and
        # cite it as if the pair shared it.
        enz = rng.choice(_CYP_PREFIXES)
        ids = [
            f"enzyme_{enz.upper()}",
            f"pk_flag_{enz}_inhibitor_A",
            f"pk_flag_{enz}_substrate_B",
        ]
        return ids, "PK_Metabolism"
    elif gold_family in _RISK_FAMILIES:
        enz = rng.choice(_CYP_PREFIXES)
        ids = [f"enzyme_{enz.upper()}", f"pk_flag_{enz}_inhibitor_A",
               f"pk_flag_{enz}_substrate_B"]
        return ids, "PK_Metabolism"
    return None


def _adv_enzyme_swap(
    pair_id: str, a_id: str, b_id: str, true_prefix: str | None,
    rng: random.Random,
) -> tuple[list[str], str] | None:
    """For PK pairs with a known true prefix, cite an unrelated enzyme."""
    if not true_prefix:
        return None
    alt_prefixes = [p for p in _CYP_PREFIXES if p != true_prefix]
    if not alt_prefixes:
        return None
    fake = rng.choice(alt_prefixes)
    ids = [
        f"enzyme_{fake.upper()}",
        f"pk_flag_{fake}_inhibitor_A",
        f"pk_flag_{fake}_substrate_B",
    ]
    # Family stays the same (still PK_Metabolism) but the mechanism is wrong.
    return ids, "PK_Metabolism"


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",     required=True)
    ap.add_argument("--pk",         required=True)
    ap.add_argument("--signatures", required=True)
    ap.add_argument("--proteins",   required=True)
    ap.add_argument("--pathways",   default=None)
    ap.add_argument("--splits",     default=None)
    ap.add_argument("--split_section", default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--output",     required=True)
    ap.add_argument("--report",     default=None)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--emit_null_ctx", action="store_true",
                    help="Also emit a null-ctx record per pair for RIS baseline decomposition.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    pk = _load_pk(Path(args.pk))
    print(f"[adv] PK flags for {len(pk):,} drugs")
    proteins = _load_drug_proteins(Path(args.proteins))
    print(f"[adv] proteins for {len(proteins):,} drugs")
    pathways = _load_drug_pathways(Path(args.pathways)) if args.pathways else {}
    print(f"[adv] pathways for {len(pathways):,} drugs")
    signatures = _load_signatures(Path(args.signatures))
    print(f"[adv] pair signatures: {len(signatures):,}")
    split_pairs = _load_split_pairs(Path(args.splits), args.split_section) if args.splits else None
    if split_pairs is not None:
        print(f"[adv] split filter: {len(split_pairs):,} pair_ids")

    labels_tbl = pq.read_table(args.labels, columns=[
        "pair_id", "a_id", "b_id", "family", "subtype",
        "subject_drugbank_id", "bidirectional",
    ]).to_pylist()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    strategy_counts: Counter = Counter()
    skipped: Counter = Counter()

    with open(args.output, "w") as f:
        for row in labels_tbl:
            n_in += 1
            pair_id = row["pair_id"]
            if split_pairs is not None and pair_id not in split_pairs:
                continue

            gold_family = row.get("family") or "unknown"
            a_id = row["a_id"]; b_id = row["b_id"]
            a_flags = pk.get(a_id, {})
            b_flags = pk.get(b_id, {})
            a_proteins = proteins.get(a_id, [])
            b_proteins = proteins.get(b_id, [])
            a_pathways = pathways.get(a_id, [])
            b_pathways = pathways.get(b_id, [])

            # True-evidence reconstruction
            if gold_family in _PK_FAMILIES:
                true_ids, true_prefix = _pk_true_ids(pair_id, a_id, b_id, a_flags, b_flags)
            elif gold_family in _PD_FAMILIES:
                true_ids = _pd_true_ids(a_id, b_id, a_proteins, b_proteins,
                                        a_pathways, b_pathways)
                true_prefix = None
            else:
                true_ids = _pd_true_ids(a_id, b_id, a_proteins, b_proteins,
                                        a_pathways, b_pathways)
                true_prefix = None
            if not true_ids:
                skipped[f"no_true_ev:{gold_family}"] += 1
                continue

            # Try enzyme_swap first for PK pairs, else cross_family
            adv = None
            strategy = None
            if gold_family in _PK_FAMILIES and true_prefix:
                adv = _adv_enzyme_swap(pair_id, a_id, b_id, true_prefix, rng)
                strategy = "enzyme_swap"
            if adv is None:
                adv = _adv_cross_family_swap(
                    pair_id, a_id, b_id, gold_family,
                    a_proteins, b_proteins, a_pathways, b_pathways, rng,
                )
                strategy = "cross_family_swap"
            if adv is None:
                skipped[f"no_adv:{gold_family}"] += 1
                continue
            adv_ids, adv_target = adv

            rec = {
                "pair_id":          pair_id,
                "a_id":             a_id,
                "b_id":             b_id,
                "gold_family":      gold_family,
                "gold_subtype":     row.get("subtype"),
                "true_context_ids": true_ids,
                "adv_context_ids":  adv_ids,
                "adv_target_family": adv_target,
                "adv_strategy":      strategy,
                "split":             args.split_section,
            }
            f.write(json.dumps(rec) + "\n")
            n_out += 1
            strategy_counts[strategy] += 1

            if args.emit_null_ctx:
                null_rec = dict(rec)
                null_rec["adv_context_ids"] = []
                null_rec["adv_target_family"] = None
                null_rec["adv_strategy"] = "null_ctx"
                f.write(json.dumps(null_rec) + "\n")
                strategy_counts["null_ctx"] += 1

    print(f"[adv] input rows: {n_in:,}")
    print(f"[adv] emitted   : {n_out:,}")
    print(f"[adv] strategies: {dict(strategy_counts)}")
    print(f"[adv] skipped   : {dict(skipped)}")

    if args.report:
        rep = Path(args.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        with open(rep, "w") as f:
            f.write("# Adversarial RIS dataset\n\n")
            f.write(f"- Source labels: `{args.labels}`\n")
            f.write(f"- Split: `{args.split_section}` ({args.splits or 'all'})\n")
            f.write(f"- Records emitted: **{n_out:,}**\n\n")
            f.write("## Strategies\n\n| strategy | n |\n|---|---:|\n")
            for k, v in strategy_counts.most_common():
                f.write(f"| `{k}` | {v:,} |\n")
            if skipped:
                f.write("\n## Skipped reasons\n\n| reason | n |\n|---|---:|\n")
                for k, v in sorted(skipped.items()):
                    f.write(f"| `{k}` | {v:,} |\n")
        print(f"[adv] wrote report {rep}")


if __name__ == "__main__":
    main()
