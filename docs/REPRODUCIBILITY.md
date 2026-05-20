# Reproducibility Notes

This document records the conventions used throughout the repository to
make the pipeline reproducible end-to-end.

## Deterministic random seeds

A single base seed lives in `configs/base.yaml -> project.seed` (default
`42`). Each module that draws random numbers receives the seed through
explicit CLI arguments rather than reading it implicitly, so a single
top-level configuration change propagates to all stages.

When multiple teachers / runs are involved, the seeds are decorrelated
by adding a known offset, e.g. teacher seeds are
`42`, `10042`, `20042` (see
`scripts/cluster_examples/run_teacher.sh`). Mirror-pair sampling and the
DPO preference shuffler use independent generators seeded from the same
base, so the per-pair ordering of (AB, BA) is fixed across reruns.

## Artifact freezing

`src.audit.freeze_phase_a` writes a SHA-256 manifest of every
data_processed parquet, JSONL, and metadata file at the end of Phase A.
Subsequent steps check that hash before consuming the file, so a silent
re-run of the data builders that changes a single byte will be
detected.

The PRM rubric (`configs/prm_rubric.yaml`) has its own SHA recorded in
`outputs/audit/prm_rubric_sha256.txt`; the teacher generator embeds
this hash inside every emitted record so traces can be traced back to
the rubric version they were produced under.

## Splits

Three splits are constructed once per DrugBank release and then frozen:

- `random_full` (80 / 10 / 10 over all pairs)
- `drug_cold`  (80 / 10 / 10 over **drugs**; no drug appears in both
  train and test)
- `pair_cold`  (80 / 10 / 10 over pairs with no drug overlap between
  train and test)

Plus a stratified `subset25k` development pool used for the bulk of
teacher generation and student SFT.

## Evaluation harness

`src.evaluation.run_full_eval` is the single entry point that produces
the full results table. It records its random seed, the hashes of the
files it consumed, the git SHA of the running checkout, and the wall
clock of every metric pass to `outputs/results/<run_name>/run_meta.json`.

All bootstrap confidence intervals use the same `n_resamples=10_000`,
paired across runs (`src.evaluation.paired_bootstrap_significance`), so
significance tests are directly comparable across rows of the main
table.

## Environment

The example cluster launchers pin the entire software stack via
`scripts/cluster_examples/setup_env.sh`. Module versions are loaded
through the cluster module system, and the pinned wheel versions for
training and evaluation live in `requirements.txt`. On a fresh node:

```bash
bash scripts/cluster_examples/setup_env.sh
source scripts/cluster_examples/activate_env.sh
python -c "import torch, transformers, peft, trl, vllm; print('ok')"
```

## Closed-weight frontier comparisons

Frontier-LLM numbers (GPT-4o, Claude Opus 4, Gemini 2.5 Pro) are
recorded together with the API model id, the request timestamp, and a
temperature of `0.0`. The exact prompts are saved next to the
predictions for audit. See `scripts/examples/run_frontier_chain.sh`.

Closed-weight model behavior is not guaranteed to be stable across API
updates; consult the recorded model id and the response timestamp when
interpreting differences across reruns.
