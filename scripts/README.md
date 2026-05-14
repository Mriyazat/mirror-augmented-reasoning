# `scripts/` — pipeline runners

Thin shell wrappers around the per-phase `python -m` entry points. Each script `cd`s to the repo root, sets `PYTHONPATH`, prints banners between steps, and supports a few env-var overrides.

| Script | Phase | What it does |
|---|---|---|
| `download_models.py`    | —       | One-shot HuggingFace snapshot of every model the pipeline uses (teachers, PRM base, student). Supports `--only <alias>` for partial downloads. |
| `run_phase_a.sh`        | A       | Parse DrugBank → parquet · pathway / PK enrichment · taxonomy · splits · audits. |
| `run_phase_b.sh`        | B       | DDI-PRM training · candidate generation per teacher · rule QC · PRM critic · cross-LLM consensus · reasoning-safety filter · preference corpora. |
| `run_phase_c.sh`        | C       | Tier-weighted SFT (+ symmetry-KL) · PRM-weighted DPO (+ hard-negatives) · optional classifier head. |
| `run_phase_d.sh`        | D       | JSON-constrained inference · conformal abstention · XGBoost reference · 8-metric eval · stress sets. |
| `run_all.sh`            | A → D   | Convenience driver chaining the four phase scripts. |

## Quick examples

```bash
# A. fetch the student base only
python scripts/download_models.py --only student

# B. run only the Llama teacher on the balanced subset
TEACHERS="llama-3.3-70b" SPLIT="subset" bash scripts/run_phase_b.sh

# C. re-run just the DPO stage with a different accelerate config
STAGES="c2" ACCELERATE_CONFIG="configs/accelerate_fsdp.yaml" \
    bash scripts/run_phase_c.sh

# D. evaluate a trained checkpoint on just the hardest split
CHECKPOINT=outputs/student/dpo/<run-name> \
SPLITS="pair_cold" \
    bash scripts/run_phase_d.sh
```

## Notes

- Every script is generic — there are **no** cluster account names, SLURM headers, or hard-coded user paths. Run them inside any GPU allocation that already has the venv activated.
- Phase B's `run_phase_b.sh` expects a running **vLLM** server reachable at `$OPENAI_API_BASE` (default `http://localhost:8000/v1`). Bring up a server per teacher checkpoint before running.
- Distributed runs use the configs under `configs/accelerate_fsdp*.yaml`. Adjust them to match the number / type of GPUs on your machine.
