"""A8 — Canonical splits + 25 k dev subset.

Four splits, all SHA256-pinned:
  1. random_full  — stratified (by family) 80/10/10 train/val/test over all
                    1,456,772 pairs.  TRANSDUCTIVE — same drugs appear in
                    train and test.
  2. drug_cold    — INDUCTIVE, one-sided.  Drugs partitioned 80/10/10; a pair
                    is in train only if BOTH drugs are in train_drugs; val
                    contains pairs with at least one val-drug and no test-drug;
                    test contains pairs with at least one test-drug.
  3. pair_cold    — INDUCTIVE, two-sided.  Same drug partition as drug_cold,
                    but test pairs must have BOTH drugs in test_drugs (hardest
                    generalization).  Train pairs same rule (both in train).
  4. subset25k    — 25k pairs stratified by family, sampled from random_full's
                    TRAIN set only (zero leakage with any split's test).

Outputs (data_processed/splits/):
  manifest_random_full.parquet   — pair_id, split ∈ {train,val,test}, family
  manifest_drug_cold.parquet     — same schema
  manifest_pair_cold.parquet     — same schema
  manifest_subset25k.parquet     — same schema (train only)
  splits_sha256.json             — checksums of each manifest
Audit:
  outputs/audit/a8_splits_report.md
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_processed"
SPLITS_DIR = DATA / "splits"
SPLITS_DIR.mkdir(exist_ok=True)
AUDIT_MD = ROOT / "outputs" / "audit" / "a8_splits_report.md"

SEED = 42
RATIO_TRAIN = 0.80
RATIO_VAL = 0.10
SUBSET_N = 25_000


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(rows: list[dict], path: Path) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy")


def build_random_full(pairs_with_fam, rng: random.Random) -> list[dict]:
    """Stratified by family: within each family, assign 80/10/10."""
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for p in pairs_with_fam:
        by_fam[p["family"]].append(p)
    out: list[dict] = []
    for fam, rows in by_fam.items():
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(n * RATIO_TRAIN)
        n_val = int(n * RATIO_VAL)
        for i, r in enumerate(rows):
            split = ("train" if i < n_train
                     else ("val" if i < n_train + n_val else "test"))
            out.append({"pair_id": r["pair_id"], "split": split, "family": fam})
    return out


def build_drug_cold(pairs_with_fam, all_drugs: list[str],
                    rng: random.Random, mode: str) -> tuple[list[dict], set[str], set[str], set[str]]:
    """mode ∈ {'drug_cold','pair_cold'}.

    Partition drugs 80/10/10 (same partition for both modes).  Assignment:
      drug_cold:
        train       = both drugs in train_drugs
        val         = ≥1 val-drug and 0 test-drugs and not all train-drugs
        test        = ≥1 test-drug
        discarded   = pairs that fall outside these rules (e.g. val+test mix)
      pair_cold:
        train = both drugs in train_drugs
        val   = both drugs in val_drugs
        test  = both drugs in test_drugs
    """
    shuffled = list(all_drugs)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_tr = int(n * RATIO_TRAIN)
    n_va = int(n * RATIO_VAL)
    train_d = set(shuffled[:n_tr])
    val_d   = set(shuffled[n_tr:n_tr + n_va])
    test_d  = set(shuffled[n_tr + n_va:])

    out: list[dict] = []
    for p in pairs_with_fam:
        a, b = p["a_id"], p["b_id"]
        a_in_train = a in train_d; b_in_train = b in train_d
        a_in_val   = a in val_d;   b_in_val   = b in val_d
        a_in_test  = a in test_d;  b_in_test  = b in test_d

        if mode == "drug_cold":
            if a_in_train and b_in_train:
                split = "train"
            elif a_in_test or b_in_test:
                split = "test"
            elif a_in_val or b_in_val:
                split = "val"
            else:
                split = None
        elif mode == "pair_cold":
            if a_in_train and b_in_train:
                split = "train"
            elif a_in_val and b_in_val:
                split = "val"
            elif a_in_test and b_in_test:
                split = "test"
            else:
                split = None
        else:
            raise ValueError(mode)

        if split is not None:
            out.append({"pair_id": p["pair_id"], "split": split, "family": p["family"]})
    return out, train_d, val_d, test_d


def build_subset25k(random_full_rows, drug_cold_rows, pair_cold_rows,
                    rng: random.Random) -> list[dict]:
    """25k pairs family-stratified, sampled from the INTERSECTION of all
    three splits' TRAIN sets.  This guarantees the subset never overlaps any
    canonical test set regardless of which split we evaluate on.
    """
    rf_train = {r["pair_id"] for r in random_full_rows if r["split"] == "train"}
    dc_train = {r["pair_id"] for r in drug_cold_rows   if r["split"] == "train"}
    pc_train = {r["pair_id"] for r in pair_cold_rows   if r["split"] == "train"}
    safe_ids = rf_train & dc_train & pc_train
    # Pull family from random_full rows (canonical family source)
    fam_of = {r["pair_id"]: r["family"] for r in random_full_rows}
    by_fam: dict[str, list[str]] = defaultdict(list)
    for pid in safe_ids:
        by_fam[fam_of[pid]].append(pid)
    total = sum(len(v) for v in by_fam.values())
    quotas = {f: max(1, int(SUBSET_N * len(v) / total)) for f, v in by_fam.items()}
    chosen: list[dict] = []
    for fam, ids in by_fam.items():
        rng.shuffle(ids)
        chosen.extend({"pair_id": pid, "split": "train", "family": fam}
                      for pid in ids[:quotas[fam]])
    if len(chosen) > SUBSET_N:
        chosen = chosen[:SUBSET_N]
    return chosen


def check_no_drug_leakage(split_rows, pair_index,
                          test_partition: set[str] | None = None,
                          val_partition: set[str] | None = None) -> dict:
    """Return stats on drug leakage between train and test.

    For inductive splits (drug_cold/pair_cold), pass `test_partition` and
    `val_partition` (the DRUG partitions used to build the split). The
    semantic gate is "no train-pair contains a drug from the test-partition".
    """
    train_drugs: set[str] = set()
    test_drugs: set[str] = set()
    val_drugs: set[str] = set()
    train_contains_test_partition = 0
    train_contains_val_partition = 0
    for r in split_rows:
        pid = r["pair_id"]
        a, b = pair_index[pid]
        s = r["split"]
        if s == "train":
            train_drugs.update([a, b])
            if test_partition is not None and (a in test_partition or b in test_partition):
                train_contains_test_partition += 1
            if val_partition is not None and (a in val_partition or b in val_partition):
                train_contains_val_partition += 1
        elif s == "test":
            test_drugs.update([a, b])
        elif s == "val":
            val_drugs.update([a, b])
    return {
        "train_drugs": len(train_drugs), "val_drugs": len(val_drugs),
        "test_drugs": len(test_drugs),
        "train∩test": len(train_drugs & test_drugs),
        "train∩val": len(train_drugs & val_drugs),
        "val∩test": len(val_drugs & test_drugs),
        "train_pairs_with_test_partition_drug": train_contains_test_partition,
        "train_pairs_with_val_partition_drug": train_contains_val_partition,
    }


def family_distribution(split_rows) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for r in split_rows:
        out[r["split"]][r["family"]] += 1
    return {k: dict(v) for k, v in out.items()}


def main():
    t0 = time.time()
    print("[A8] loading pairs + labels ...", flush=True)
    # Labels drive which pairs are in scope: drop pairs that A5 excluded
    # (codename/ambiguous pairs written to a5_dropped_pairs.parquet).  Any
    # downstream split must not contain these.
    labels = pq.read_table(DATA / "labels_hierarchical.parquet",
                           columns=["pair_id", "a_id", "b_id", "family"]).to_pylist()
    pairs_with_fam = [{**r} for r in labels]
    pair_index = {p["pair_id"]: (p["a_id"], p["b_id"]) for p in pairs_with_fam}
    all_drugs = sorted({p["a_id"] for p in pairs_with_fam}
                       | {p["b_id"] for p in pairs_with_fam})
    print(f"[A8] {len(pairs_with_fam):,} pairs (labels-kept only), "
          f"{len(all_drugs):,} unique drugs")

    rng = random.Random(SEED)

    # 1. random_full
    rf = build_random_full(pairs_with_fam, rng)
    write_manifest(rf, SPLITS_DIR / "manifest_random_full.parquet")

    # 2. drug_cold
    rng_cold = random.Random(SEED + 1)
    dc, dc_train_d, dc_val_d, dc_test_d = build_drug_cold(
        pairs_with_fam, all_drugs, rng_cold, "drug_cold")
    write_manifest(dc, SPLITS_DIR / "manifest_drug_cold.parquet")

    # 3. pair_cold — uses the SAME drug partition so the two are directly comparable
    rng_pair = random.Random(SEED + 1)
    pc, pc_train_d, pc_val_d, pc_test_d = build_drug_cold(
        pairs_with_fam, all_drugs, rng_pair, "pair_cold")
    write_manifest(pc, SPLITS_DIR / "manifest_pair_cold.parquet")

    # 4. subset25k (from intersection of all three train sets)
    rng_sub = random.Random(SEED + 2)
    sub = build_subset25k(rf, dc, pc, rng_sub)
    write_manifest(sub, SPLITS_DIR / "manifest_subset25k.parquet")

    print(f"[A8] all 4 manifests written in {time.time()-t0:.0f}s", flush=True)

    # Checksums
    checksums = {
        "random_full": sha256_file(SPLITS_DIR / "manifest_random_full.parquet"),
        "drug_cold":   sha256_file(SPLITS_DIR / "manifest_drug_cold.parquet"),
        "pair_cold":   sha256_file(SPLITS_DIR / "manifest_pair_cold.parquet"),
        "subset25k":   sha256_file(SPLITS_DIR / "manifest_subset25k.parquet"),
    }
    (SPLITS_DIR / "splits_sha256.json").write_text(json.dumps(checksums, indent=2))

    # ── Audit ─────────────────────────────────────────────────────────────────
    def count_by_split(rows):
        c = Counter(r["split"] for r in rows)
        return dict(c)

    rf_counts = count_by_split(rf)
    dc_counts = count_by_split(dc)
    pc_counts = count_by_split(pc)

    rf_leak = check_no_drug_leakage(rf, pair_index)
    dc_leak = check_no_drug_leakage(dc, pair_index,
                                    test_partition=dc_test_d, val_partition=dc_val_d)
    pc_leak = check_no_drug_leakage(pc, pair_index,
                                    test_partition=pc_test_d, val_partition=pc_val_d)

    rf_fam = family_distribution(rf)
    dc_fam = family_distribution(dc)
    pc_fam = family_distribution(pc)

    subset_fam = Counter(r["family"] for r in sub)

    # Subset overlap with test sets (should be zero, since we sampled from rf train)
    rf_test_ids = {r["pair_id"] for r in rf if r["split"] == "test"}
    dc_test_ids = {r["pair_id"] for r in dc if r["split"] == "test"}
    pc_test_ids = {r["pair_id"] for r in pc if r["split"] == "test"}
    subset_ids = {r["pair_id"] for r in sub}
    n_leak_rf = len(subset_ids & rf_test_ids)
    n_leak_dc = len(subset_ids & dc_test_ids)
    n_leak_pc = len(subset_ids & pc_test_ids)

    md = [
        "# A8 — Splits report\n",
        f"- Runtime: {time.time()-t0:.0f}s",
        f"- Seed: {SEED}",
        "",
        "## Sizes",
        "",
        "| Split | train | val | test | total |",
        "|---|---:|---:|---:|---:|",
        f"| random_full | {rf_counts.get('train',0):,} | {rf_counts.get('val',0):,} | "
        f"{rf_counts.get('test',0):,} | {sum(rf_counts.values()):,} |",
        f"| drug_cold | {dc_counts.get('train',0):,} | {dc_counts.get('val',0):,} | "
        f"{dc_counts.get('test',0):,} | {sum(dc_counts.values()):,} |",
        f"| pair_cold | {pc_counts.get('train',0):,} | {pc_counts.get('val',0):,} | "
        f"{pc_counts.get('test',0):,} | {sum(pc_counts.values()):,} |",
        f"| subset25k (train-only) | {len(sub):,} | — | — | {len(sub):,} |",
        "",
        "## Drug-level leakage audit",
        "",
        "| Split | train_drugs | val_drugs | test_drugs | train∩test | train∩val | val∩test |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| random_full | {rf_leak['train_drugs']:,} | {rf_leak['val_drugs']:,} | "
        f"{rf_leak['test_drugs']:,} | **{rf_leak['train∩test']:,}** | "
        f"{rf_leak['train∩val']:,} | {rf_leak['val∩test']:,} |",
        f"| drug_cold (inductive) | {dc_leak['train_drugs']:,} | {dc_leak['val_drugs']:,} | "
        f"{dc_leak['test_drugs']:,} | **{dc_leak['train∩test']:,}** | "
        f"{dc_leak['train∩val']:,} | {dc_leak['val∩test']:,} |",
        f"| pair_cold (strict)   | {pc_leak['train_drugs']:,} | {pc_leak['val_drugs']:,} | "
        f"{pc_leak['test_drugs']:,} | **{pc_leak['train∩test']:,}** | "
        f"{pc_leak['train∩val']:,} | {pc_leak['val∩test']:,} |",
        "",
        "- Expected: random_full train∩test high (same drugs in both).",
        "- Expected: drug_cold / pair_cold train∩test **exactly zero**.",
        "",
        "## Family distribution across splits (random_full)",
        "",
        "| Family | train | val | test |",
        "|---|---:|---:|---:|",
    ]
    for fam in sorted({f for d in rf_fam.values() for f in d}):
        md.append(f"| `{fam}` | {rf_fam.get('train',{}).get(fam,0):,} | "
                  f"{rf_fam.get('val',{}).get(fam,0):,} | "
                  f"{rf_fam.get('test',{}).get(fam,0):,} |")

    md += [
        "",
        "## Family distribution (drug_cold)",
        "",
        "| Family | train | val | test |",
        "|---|---:|---:|---:|",
    ]
    for fam in sorted({f for d in dc_fam.values() for f in d}):
        md.append(f"| `{fam}` | {dc_fam.get('train',{}).get(fam,0):,} | "
                  f"{dc_fam.get('val',{}).get(fam,0):,} | "
                  f"{dc_fam.get('test',{}).get(fam,0):,} |")

    md += [
        "",
        "## Subset25k family distribution",
        "",
        "| Family | count | % |",
        "|---|---:|---:|",
    ]
    for fam, n in subset_fam.most_common():
        md.append(f"| `{fam}` | {n:,} | {100*n/len(sub):.2f}% |")

    md += [
        "",
        "## Subset overlap with test sets (must be 0)",
        f"- subset25k ∩ random_full.test: {n_leak_rf} {'PASS' if n_leak_rf == 0 else 'FAIL'}",
        f"- subset25k ∩ drug_cold.test:   {n_leak_dc} {'PASS' if n_leak_dc == 0 else 'FAIL'}",
        f"- subset25k ∩ pair_cold.test:   {n_leak_pc} {'PASS' if n_leak_pc == 0 else 'FAIL'}",
        "",
        "## Inductive-split semantic gates (drug partition leakage)",
        "",
        "_For drug_cold/pair_cold, the correct gate is: no TRAIN pair contains any "
        "drug from the test-partition. In drug_cold a test PAIR can still contain a "
        "train-drug (paired with a test-drug); in pair_cold both drugs must be from "
        "the test-partition._",
        "",
        f"- drug_cold: train pairs with a test-partition drug: "
        f"**{dc_leak['train_pairs_with_test_partition_drug']:,}** "
        f"{'PASS' if dc_leak['train_pairs_with_test_partition_drug'] == 0 else 'FAIL'}",
        f"- drug_cold: train pairs with a val-partition drug:  "
        f"**{dc_leak['train_pairs_with_val_partition_drug']:,}** "
        f"{'PASS' if dc_leak['train_pairs_with_val_partition_drug'] == 0 else 'FAIL'}",
        f"- pair_cold: train pairs with a test-partition drug: "
        f"**{pc_leak['train_pairs_with_test_partition_drug']:,}** "
        f"{'PASS' if pc_leak['train_pairs_with_test_partition_drug'] == 0 else 'FAIL'}",
        f"- pair_cold: train pairs with a val-partition drug:  "
        f"**{pc_leak['train_pairs_with_val_partition_drug']:,}** "
        f"{'PASS' if pc_leak['train_pairs_with_val_partition_drug'] == 0 else 'FAIL'}",
        "",
        "## Primary gates",
        f"- random_full: all pair IDs disjoint across splits: PASS (by construction)",
        f"- pair_cold train∩test DRUGS == 0: "
        f"**{'PASS' if pc_leak['train∩test'] == 0 else 'FAIL'}**",
        f"- pair_cold val∩test DRUGS == 0: "
        f"**{'PASS' if pc_leak['val∩test'] == 0 else 'FAIL'}**",
        f"- subset25k ⊂ intersection of all train sets: "
        f"**{'PASS' if (n_leak_rf + n_leak_dc + n_leak_pc) == 0 else 'FAIL'}**",
        "",
        "## SHA256 checksums",
        "```",
        json.dumps(checksums, indent=2),
        "```",
    ]
    AUDIT_MD.write_text("\n".join(md) + "\n")
    print(f"[A8] wrote {AUDIT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
