<h1 align="center">MARD</h1>

<p align="center">
  <b>Mirror-Augmented Reasoning Distillation for<br>
  Mechanism-Level Drug–Drug Interaction Prediction</b>
</p>

<p align="center">
  <i>Research code accompanying our EMNLP 2026 submission.</i>
</p>

<p align="center">
  <a href="LICENSE"><img alt="Code license" src="https://img.shields.io/badge/code-MIT-blue.svg"></a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img alt="Adapters license" src="https://img.shields.io/badge/adapters%20%26%20corpora-CC%20BY--NC%204.0-lightgrey.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white">
  <img alt="Venue" src="https://img.shields.io/badge/venue-EMNLP%202026-9d174d.svg">
  <a href="https://go.drugbank.com/"><img alt="DrugBank" src="https://img.shields.io/badge/data-DrugBank-2c8e3f.svg"></a>
</p>

<p align="center">
  <a href="#abstract">Abstract</a> ·
  <a href="#pipeline-overview">Pipeline</a> ·
  <a href="#end-to-end-case-study-for-pair-db00582--db06626-voriconazole--axitinib">Case study</a> ·
  <a href="#repository-layout">Layout</a> ·
  <a href="#installation">Install</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#metrics">Metrics</a> ·
  <a href="#license">License</a>
</p>

---

## Abstract

<div align="justify">

Mechanism-level drug–drug interaction (DDI) prediction requires
identifying <i>which</i> enzyme or pharmacodynamic axis is implicated,
in <i>which</i> direction, and with <i>which</i> evidence — not merely
whether two drugs interact. We introduce a reproducible mechanism-level
DDI labelling and evaluation protocol with a structured
7-family / 147-subtype taxonomy, leakage-safe cold-split protocols, and
auditable reasoning metrics for evaluating pharmacological prediction
beyond flat interaction classification. We propose a pipeline that
produces a 7B reasoning <b>MARD</b>
(<b>M</b>irror-<b>A</b>ugmented <b>R</b>easoning <b>D</b>istillation),
combining three training innovations: a single-token KL on the
direction tag that ties the model's prediction, per-loss PRM-weighted
DPO with programmatic hard negatives, and a leakage-safe
mechanism-aware retrieval channel. Process-reward step labels are
automatically verifiable against DrugBank-structured fields, requiring
no human annotation or LLM judge. On the April-2026 DrugBank release,
our <b>MARD-7B</b> is the only system in a 32-system comparison whose
accuracy survives drug-pair novelty, beating the best baseline by
<b>+13.9 pp</b> and GPT-4o by <b>+6.7 pp</b> at ~1% of frontier API
cost. Further analysis reveals an <i>anti-memorisation</i> signature
where accuracy improves on rarely seen drugs, suggesting that gain
comes from structured pharmacological reasoning rather than
drug-frequency memorisation.

</div>

## Pipeline overview

<div align="justify">

The pipeline produces a small, calibrated DDI predictor that emits
mechanism-grounded reasoning traces. At a high level it has four
phases:

</div>

- **Phase A — Data construction.** A DrugBank-derived corpus, a
  curated 7-family / multi-subtype label taxonomy with directionality,
  pair-level mechanistic signatures, and three reproducible
  train/val/test splits (random, drug-cold, pair-cold) plus a
  stratified development subset.
- **Phase B — Teacher generation and PRM training.** A heterogeneous
  three-teacher ensemble (Llama-3.3-70B, Qwen2.5-72B,
  DeepSeek-R1-Distill-Llama-70B) produces step-structured reasoning
  traces. A Process Reward Model is then trained on auto-verifiable
  step-level signals (evidence grounding, direction preservation,
  family consistency, PK-flag consistency, and others) to filter and
  re-rank candidate traces.
- **Phase C — Student training.** A 7B student is fine-tuned on the
  PRM-filtered teacher traces with a faithfulness loss and a
  mirror-pair symmetry-KL term, followed by preference optimization on
  hard-negative and direction-mirror pairs.
- **Phase D — Evaluation.** Standard classification metrics together
  with the novel reasoning-faithfulness metrics introduced in this
  work, conformal abstention, verifier re-ranking, and head-to-head
  comparisons against frontier LLMs and DDI-specific baselines.

<div align="justify">

This repository is the source code, configuration, and example
launchers that produced the results in the paper.

</div>

## End-to-end case study for pair DB00582 | DB06626 (Voriconazole + Axitinib)

<div align="justify">

A concrete illustration of what MARD-7B does on a single DrugBank
pair. Both panels are discussed in the paper (Fig.&nbsp;3 /
Appendix&nbsp;A); they are reproduced here so the
input → reasoning → verifiable-output flow is visible without opening
the PDF.

</div>

### Pipeline at a glance

<div align="justify">

The five-stage view from the structured input pool, through the
mirror-tied SFT and PRM-weighted DPO objectives, to the
schema-constrained reasoning trace and verified final answer:

</div>

<p align="center">
  <img src="docs/figures/case_study_overview.png"
       alt="Five-stage MARD pipeline on the Voriconazole + Axitinib pair: structured drug-pair input, evidence pool with PK flags and retrieved neighbours, mirror-tied SFT with KL on the direction tag, schema-constrained reasoning trace, and verified output checked against DrugBank."
       width="100%">
</p>

### What the model actually reads and emits

<div align="justify">

A drill-down on the same pair: the raw DrugBank fields, the PK-flag
table and pair-level similarity scalars the model receives, the five
retrieved labelled neighbours, the four-step reasoning trace, and the
structured prediction
(<code>PK_Metabolism / metabolism / a_to_b / down</code>,
confidence 0.85). Every cited identifier appears verbatim in the
evidence pool, so each step is independently checkable against
DrugBank without any LLM judge in the loop:

</div>

<p align="center">
  <img src="docs/figures/case_study_voriconazole_axitinib.png"
       alt="Detailed case study panel for DB00582|DB06626 showing raw DrugBank ATC / enzymes / proteins / pathway fields for Voriconazole and Axitinib, the PK-flag table the model sees, four pair-level similarity scalars, K=5 retrieved labelled neighbours, the four-step reasoning trace (pk_flag, protein, neighbor, conclusion), and the structured prediction with calibrated confidence."
       width="100%">
</p>

## Repository Layout

```
EMNLP2026_DDI_Verifier_Release/
├── configs/                  # YAML configs (base + PRM rubric + FSDP)
├── docs/
│   ├── figures/              # Case-study figures used in README
│   ├── PIPELINE_OVERVIEW.md  # End-to-end pipeline reference
│   └── REPRODUCIBILITY.md    # Hashes, configs, seed protocol
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

<div align="justify">

For GPU clusters, portable example launchers live under
<code>scripts/cluster_examples/</code>. They target a typical
4&nbsp;×&nbsp;H100 80&nbsp;GB node and read <code>--account</code>,
venv path, and module names from environment variables (see
<code>scripts/cluster_examples/setup_env.sh</code> and
<code>scripts/cluster_examples/activate_env.sh</code>).

</div>

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

<div align="justify">

The pipeline operates on a fixed [DrugBank][drugbank] XML release
(the exact version is recorded in
<code>configs/base.yaml&nbsp;-&gt;&nbsp;data_sources.drugbank</code>)
together with DDInter severity metadata (used only as metadata, never
as a prediction target) and KEGG / SMPDB pathway maps.

Raw redistributable data is <b>not</b> included here. After obtaining
DrugBank (license required) and the pathway dumps, the
data-construction modules in <code>src/data/</code> reconstruct the
canonical processed parquets and JSONL splits. The exact file paths,
expected counts, and SHA-256 hashes are pinned in
<code>configs/base.yaml</code>.

</div>

## Evaluation

<div align="justify">

Headline numbers are produced by <code>src.evaluation.run_full_eval</code>,
which takes a predictions JSONL plus a ground-truth manifest and
writes the full metrics table together with bootstrap confidence
intervals to <code>outputs/results/&lt;run_name&gt;/</code>.

</div>

```bash
python -m src.evaluation.run_full_eval \
    --predictions outputs/predictions/student_rerank.jsonl \
    --labels      data_processed/labels_hierarchical.parquet \
    --split       random_full \
    --run_name    student_rerank
```

<div align="justify">

Frontier-LLM comparison runs (GPT-4o, Claude Opus 4, Gemini 2.5 Pro)
are launched through
<code>scripts/examples/run_frontier_chain.sh</code>, which records the
exact API model id and request parameters next to the predictions.

</div>

## Metrics

<div align="justify">

Beyond macro-F1 and accuracy, we report the following metrics
(implemented in <code>src/metrics/</code> and re-exported by
<code>src/metrics/__init__.py</code>):

</div>

| Metric | Meaning                                                      |
| :----: | ------------------------------------------------------------ |
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

<div align="justify">

Bootstrap confidence intervals and paired significance tests are in
<code>src/evaluation/bootstrap_ci.py</code> and
<code>src/evaluation/paired_bootstrap_significance.py</code>.

</div>

## Tests

```bash
pytest -q
```

<div align="justify">

Unit tests cover the metrics library, conformal abstention, preference
pair construction, and the evaluation harness adapters.

</div>

## License

<div align="justify">

The <b>code</b> in this repository is released under the
[MIT License](LICENSE).

The <b>released LoRA adapters</b> (MARD-7B SFT and PRM-DPO checkpoints,
PRM scorer) and <b>curated derivative corpora</b> (mirror-augmented
SFT splits, preference pairs, evaluation manifests) will be released
under [CC BY-NC 4.0][cc-by-nc] for non-commercial research use, in
compliance with [DrugBank][drugbank]'s academic-licence terms.

The underlying <b>raw datasets</b> ([DrugBank][drugbank] XML, DDInter
severity table, KEGG / SMPDB pathway dumps) are <b>not</b>
redistributed here: each carries its own upstream licence and must be
obtained directly from the source. The pipeline reconstructs the
canonical processed files locally from a licensed [DrugBank][drugbank]
release plus the pathway dumps; the expected file paths and SHA-256
hashes are pinned in <code>configs/base.yaml</code>.

</div>

## Acknowledgments

<div align="justify">

We thank the maintainers of the open-source projects this work builds
upon, including [Med-PRM][med-prm], HuggingFace
[Transformers][hf-transformers] / [Accelerate][hf-accelerate] /
[PEFT][hf-peft] / [TRL][hf-trl], [vLLM][vllm], and the OpenDDI
baselines repository.

</div>

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[drugbank]: https://go.drugbank.com/
[med-prm]: https://github.com/dmis-lab/Med-PRM
[hf-transformers]: https://github.com/huggingface/transformers
[hf-accelerate]: https://github.com/huggingface/accelerate
[hf-peft]: https://github.com/huggingface/peft
[hf-trl]: https://github.com/huggingface/trl
[vllm]: https://github.com/vllm-project/vllm
