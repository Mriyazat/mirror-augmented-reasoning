# DDI Verifier: PRM-Guided Distillation and Verifier-Reranked Inference for Drug-Drug Interaction Prediction

Research code accompanying our EMNLP 2026 submission.

The pipeline produces a small, calibrated DDI predictor that emits
mechanism-grounded reasoning traces. At a high level it has four phases:

- **Phase A — Data construction.** A DrugBank-derived corpus, a curated
  7-family / multi-subtype label taxonomy with directionality, pair-level
  mechanistic signatures, and three reproducible train/val/test splits
  (random, drug-cold, pair-cold) plus a stratified development subset.
- **Phase B — Teacher generation and PRM training.** A heterogeneous
  three-teacher ensemble (Llama-3.3-70B, Qwen2.5-72B,
  DeepSeek-R1-Distill-Llama-70B) produces step-structured reasoning
  traces. A Process Reward Model is then trained on auto-verifiable
  step-level signals (evidence grounding, direction preservation, family
  consistency, PK-flag consistency, and others) to filter and re-rank
  candidate traces.
- **Phase C — Student training.** A 7B student is fine-tuned on the
  PRM-filtered teacher traces with a faithfulness loss and a mirror-pair
  symmetry-KL term, followed by preference optimization on hard-negative
  and direction-mirror pairs.
- **Phase D — Evaluation.** Standard classification metrics together
  with the novel reasoning-faithfulness metrics introduced in this work,
  conformal abstention, verifier re-ranking, and head-to-head
  comparisons against frontier LLMs and DDI-specific baselines.

This repository is the source code, configuration, and example launchers
that produced the results in the paper.

## Repository Layout

```
EMNLP2026_DDI_Verifier_Release/
├── configs/                  # YAML configs (base + PRM rubric + FSDP)
├── docs/                     # Pipeline overview, reproducibility notes
├── notebooks/                # Optional analysis notebooks
├── scripts/
│   ├── cluster_examples/     # Example portable SLURM launchers
│   └── examples/             # Example local commands
├── src/
│   ├── audit/                # Data-integrity audits
│   ├── data/                 # Corpus construction + split builders
│   ├── teacher/              # Teacher generation, PRM, critic, judges
│   ├── training/             # SFT + DPO + sweep tooling
│   ├── inference/            # Predict + verifier re-rank + abstention
│   ├── evaluation/           # Eval harness + bootstrap CIs + analyses
│   ├── metrics/              # Novel + standard metrics
│   └── visualization/        # Figure / table generation
├── tests/                    # Unit tests
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone <repository-url> EMNLP2026_DDI_Verifier_Release
cd EMNLP2026_DDI_Verifier_Release
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export PYTHONPATH="$PWD:$PYTHONPATH"
```

For GPU clusters, portable example launchers live under
`scripts/cluster_examples/`. They target a typical 4 x H100 80GB node
and read `--account`, venv path, and module names from environment
variables (see `scripts/cluster_examples/setup_env.sh` and
`scripts/cluster_examples/activate_env.sh`).

## Environment Variables

| Variable             | Purpose                                                  |
| -------------------- | -------------------------------------------------------- |
| `DDI_ROOT`           | Repository root (defaults to `.`)                        |
| `DDI_OUTPUTS`        | Output directory for predictions and eval artifacts      |
| `DDI_CKPT`           | Checkpoint directory for student and PRM adapters        |
| `OPENAI_API_KEY`     | Required for `--provider openai` frontier evaluations    |
| `ANTHROPIC_API_KEY`  | Required for `--provider anthropic` judges               |
| `GOOGLE_API_KEY`     | Required for `--provider google` (Gemini) judges         |

## Data

The pipeline operates on a fixed DrugBank XML release (the exact version
is recorded in `configs/base.yaml -> data_sources.drugbank`) together
with DDInter severity metadata (used only as metadata, never as a
prediction target) and KEGG / SMPDB pathway maps.

Raw redistributable data is **not** included here. After obtaining
DrugBank (license required) and the pathway dumps, the data-construction
modules in `src/data/` reconstruct the canonical processed parquets and
JSONL splits. The exact file paths, expected counts, and SHA-256 hashes
are pinned in `configs/base.yaml`.

## Evaluation

Headline numbers are produced by `src.evaluation.run_full_eval`, which
takes a predictions JSONL plus a ground-truth manifest and writes the
full metrics table together with bootstrap confidence intervals to
`outputs/results/<run_name>/`.

```bash
python -m src.evaluation.run_full_eval \
    --predictions outputs/predictions/student_rerank.jsonl \
    --labels      data_processed/labels_hierarchical.parquet \
    --split       random_full \
    --run_name    student_rerank
```

Frontier-LLM comparison runs (GPT-4o, Claude Opus 4, Gemini 2.5 Pro) are
launched through `scripts/examples/run_frontier_chain.sh`, which records
the exact API model id and request parameters next to the predictions.

## Metrics

Beyond macro-F1 and accuracy, we report the following metrics
(implemented in `src/metrics/` and re-exported by
`src/metrics/__init__.py`):

| Metric | Meaning                                                      |
| ------ | ------------------------------------------------------------ |
| MFS    | Mechanism-Faithfulness Score (rationale vs. mechanism)       |
| MPS    | Mirror-Pair Separation (direction-symmetry awareness)        |
| CFS    | Counterfactual-Faithfulness Score                            |
| RPC    | Reasoning-Prediction Coherence (rationale vs. final answer)  |
| THS    | Taxonomy-Hierarchy Score                                     |
| CSA    | Compositional-Subtype Accuracy                               |
| HR     | Hallucination Rate (rationale entities outside DrugBank)     |
| AU     | Abstention Utility (selective coverage / accuracy)           |
| SLFS   | Selective-Label Faithfulness Score                           |
| MOR    | Mechanism-Overlap Ratio (retrieval audit, optional)          |

Bootstrap confidence intervals and paired significance tests are in
`src/evaluation/bootstrap_ci.py` and
`src/evaluation/paired_bootstrap_significance.py`.

## Tests

```bash
pytest -q
```

Unit tests cover the metrics library, conformal abstention, preference
pair construction, and the evaluation harness adapters.

## License

Released under the MIT License (see `LICENSE`). DrugBank and DDInter
each have their own licenses; redistribution of those datasets from this
repository is not permitted.

## Citation

```bibtex
@inproceedings{ddi_verifier_emnlp2026,
  title     = {PRM-Guided Distillation and Verifier-Reranked Inference
               for Drug--Drug Interaction Prediction},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026}
}
```

## Acknowledgments

We thank the maintainers of the open-source projects this work builds
upon, including [Med-PRM](https://github.com/dmis-lab/Med-PRM),
HuggingFace Transformers / Accelerate / PEFT / TRL,
[vLLM](https://github.com/vllm-project/vllm),
and the OpenDDI baselines repository.
