# split construction — Subset Sampling Protocol (no external constraint)

**Goal.** Produce three evaluation splits from DrugBank 2026-04 that (a) stress-test true generalization, (b) are cheap to iterate on, and (c) are reproducible by hash. No test-set overlap concerns because splits are rebuilt from scratch.

## 0. Inputs

- `data_processed/pairs_canonical.parquet` — one row per canonical (drug_a, drug_b) pair with description, pharmacology flags, KEGG pathway signatures, DDInter severity (if any). Built in **A2-A7**.
- ~**1.46M** canonical pairs total (from pair construction). Scale is a feature, not a bug.

## 1. Fixed seeds & hash manifest

- Global seed `42`. Each split writes a `SHA256(sorted_pairs)` to `data_processed/splits/<name>/manifest.json` so anyone can recompute and bit-compare.

## 2. Three primary splits

| Name | Purpose | How we sample | Size targets |
|------|---------|---------------|-------------|
| `random_full` | Per-interaction IID benchmark (matches most prior work) | Uniform 80/10/10 on the 1.46M pairs after canonicalization | train ~1.17M · val ~146k · test ~146k |
| `drug_cold` | Test on drugs never seen at train time (true inductive) | Partition the 19,853 drugs 80/10/10; a pair is in `test` iff **both** drugs are in the test drug set. Val uses a pair with **at least one** drug in val drug set. | target test ≈ 50-100k pairs (actual shape measured from partition) |
| `pair_cold` | Drugs can overlap but the specific pair is held out (harder than random, easier than drug_cold) | Random 80/10/10 on pairs, enforcing zero pair overlap; drugs are free to repeat | train 1.17M · val 146k · test 146k |

Each split writes: `train_pairs.jsonl`, `val_pairs.jsonl`, `test_pairs.jsonl`, `manifest.json` (seed, sha256, counts, drug-overlap statistics, label histogram).

## 3. The **25k balanced subset** (fast iteration)

Used for **Phase B** (teacher generation) and **student SFT/DPO** (SFT + DPO dev runs). Frozen once.

| Property | Value |
|---|---|
| Source | `random_full` train/val/test (no leakage — we sub-sample inside existing splits) |
| Size | 20k train / 2.5k val / 2.5k test |
| Balancing | Per-label quota (capped at 500 per class); residual capacity filled uniformly to keep head/tail diversity |
| Severity coverage | Re-balanced so ≥ 30% of pairs have DDInter severity metadata so we can probe severity-stratified metrics early |
| Drug diversity | Reject sample: no more than **X%** of the 25k can reference a single drug (target X≈3%) to avoid shortcut learning on super-connected drugs |
| Hash | `data_processed/splits/subset25k/manifest.json` with `SHA256(sorted_pair_ids)` |

## 4. Scale-up protocol (Phase D)

- `100k_balanced` (intermediate) — same algorithm, quota 2k/class.
- `full_1.4M` (final) — all `random_full` pairs.
- Drug-cold + pair-cold always run on their full sizes; no subsetting.

## 5. Safety & leakage gates (automated, run at split time)

1. **Pair dedupe**: assert canonical (a,b) with a<b appears exactly once per split file.
2. **Cross-split pair leakage**: `train ∩ val == ∅` and `train ∩ test == ∅` — hard assertion.
3. **Cross-split drug leakage** (drug_cold only): `train_drugs ∩ test_drugs == ∅`.
4. **Label coverage**: every label appearing in train must appear in val+test (else flag).
5. **Severity coverage**: report %Major/%Moderate/%Minor/%Unknown per split.
6. **Re-shuffle ban**: each split writes `shuffle_seed` into manifest — recomputing with same seed must produce identical `SHA256`.

## 6. Deliverables (code in A8)

- `src/data/build_splits.py` — runs all three splits + the 25k subset, writes manifests.
- `tests/test_splits.py` — pytest assertions for gates 1-6 above.
- `outputs/audit/a08_split_stats.md` — table of sizes, label coverage, severity mix.

## 7. Decisions recorded

- **Directionality.** In each canonical pair row we store `raw_subject_id` (the drug that was the grammatical subject of the interaction text) so Phase A (taxonomy) (hierarchical taxonomy) can recover direction without re-parsing XML.
- **Cost.** We *do not* train full-scale models until A8 gates pass. Teacher generation in Phase B runs against the 25k subset first.
