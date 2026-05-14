<div align="center">

# CoT-DDI

### Chain-of-Thought Distillation for Drug–Drug Interaction Prediction

*A 7B reasoner that explains the mechanism, not just the label.*

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HF%20Transformers-4.42-yellow.svg)](https://huggingface.co/docs/transformers)
[![TRL](https://img.shields.io/badge/TRL-0.9.4-9cf.svg)](https://huggingface.co/docs/trl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)

</div>

---

## Why CoT-DDI?

Most drug–drug interaction (DDI) benchmarks predict a single top-1 label. In the clinic that is not enough — a pharmacist needs the **mechanism**: which enzyme, transporter, or pharmacodynamic axis is implicated, the **direction** of the interaction (A→B, B→A, bidirectional), the **evidence**, and a calibrated **abstention** when the evidence is thin.

CoT-DDI is a four-stage distillation pipeline that produces a **7B Qwen-2.5 student** which emits step-wise reasoning grounded in a structured evidence pool and concludes in a JSON object with `family`, `subtype`, `direction`, and an ≤80-word natural-language summary.

```
┌─────────────────┐   ┌──────────────────┐   ┌────────────────┐   ┌──────────────────┐
│  A. Data &      │ → │  B. Multi-Teacher│ → │  C. Student    │ → │  D. Evaluation   │
│  Taxonomy       │   │  Consensus +     │   │  SFT + DPO     │   │  on 3 splits +   │
│  (DrugBank)     │   │  DDI-PRM critic  │   │  + Mirror Aug. │   │  ablations       │
└─────────────────┘   └──────────────────┘   └────────────────┘   └──────────────────┘
```

### What's inside

- **Cross-teacher consensus** of 3 frontier teachers × 24 candidates/pair, scored by 10 rule-based QC gates and a fine-tuned DDI-PRM.
- **Mirror-augmented SFT** with a position-restricted symmetry-KL loss (drives mirror flip-rate from 51 % → < 5 %).
- **PRM-weighted DPO / IPO** with four families of programmatic hard-negatives.
- **Mechanism-aware retrieval** of top-K neighbour pairs at inference (ablation: removing it collapses macro-F1 from 0.797 → 0.178 on the same checkpoint).
- **8-metric evaluation suite** — macro-F1, MFS, MPS, CSA, RPC, AU, HR, THS — over three splits (`random_full`, `drug_cold`, `pair_cold`).

---

## Repository Layout

```
CoT_DDI/
├── configs/                  # YAML configs (base + per-phase accelerate/FSDP)
│   ├── base.yaml             # paths, splits, models, loss weights, metric thresholds
│   ├── accelerate_fsdp.yaml
│   ├── accelerate_fsdp_qwen.yaml
│   ├── accelerate_fsdp_prm.yaml
│   └── prm_rubric.yaml       # step-level rubric for the DDI-PRM critic
│
├── src/
│   ├── data/                 # Phase A — corpus construction
│   │   ├── parse_drugbank.py        # XML → parquet (drugs, pairs, pathways, x-refs)
│   │   ├── fetch_pathways.py        # KEGG + SMPDB pathway harvest
│   │   ├── build_pk_table.py        # CYP / transporter / PK flags per drug
│   │   ├── build_signatures.py      # pathway-Jaccard pair signatures
│   │   ├── build_taxonomy.py        # 7-family × 100-subtype DDI taxonomy
│   │   ├── build_splits.py          # random_full / drug_cold / pair_cold + subset25k
│   │   ├── build_mirror_sft_corpus.py
│   │   ├── build_adversarial.py
│   │   ├── build_counterfactual.py
│   │   ├── build_polypharmacy.py
│   │   ├── build_student_eval_prompts.py
│   │   └── prepare_phase_c.py       # tier-weighted SFT/preference corpus
│   │
│   ├── audit/                # Phase A audits + GO/NO-GO freeze
│   ├── teacher/              # Phase B — multi-teacher trace generation
│   │   ├── prompt.py, schema.py, context_builder.py, provider.py
│   │   ├── generate.py              # vLLM-backed trace generator (24 candidates/pair)
│   │   ├── qc.py                    # 10 rule-based QC gates
│   │   ├── critic.py                # DDI-PRM step-level critic
│   │   ├── prm_data.py, prm_train.py, prm_verify.py
│   │   ├── critic_rerank.py
│   │   ├── merge.py, merge_consensus.py
│   │   ├── apply_reasoning_safety.py
│   │   ├── llm_judge.py             # OpenAI / Anthropic / Gemini judge harness
│   │   ├── sample_for_judge.py
│   │   ├── audit_teacher_clean.py
│   │   ├── build_preference_pairs.py
│   │   ├── build_direction_mirror_preferences.py
│   │   ├── build_phase4_hard_negative_preferences.py
│   │   ├── drugbank_crosscheck.py
│   │   └── score_traces_prm.py
│   │
│   ├── training/             # Phase C — student training
│   │   ├── sft_train.py             # tier-weighted SFT + faithfulness + symmetry-KL
│   │   ├── dpo_mirror.py            # PRM-weighted DPO/IPO (exact + IS fallback)
│   │   ├── train_classifier_head.py
│   │   ├── evaluate_sweep.py
│   │   └── summarize_sweep.py
│   │
│   ├── inference/            # Phase D — prediction + abstention
│   │   ├── predict.py               # student inference w/ retrieval + JSON parse
│   │   ├── abstention.py            # conformal + entropy gating
│   │   └── augment_predictions.py
│   │
│   ├── evaluation/           # Phase D — eval harness + baselines
│   │   ├── run_full_eval.py         # 8-metric suite on all 3 splits
│   │   └── baseline_xgboost.py
│   │
│   └── metrics/              # The 8 metrics (see § Metrics)
│       ├── mfs.py  mps.py  csa.py  rpc.py  au.py  hr.py  ths.py  ris.py
│       ├── cfs.py  slfs.py  mor.py
│
├── tests/                    # pytest unit tests
│   ├── test_metrics.py
│   ├── test_eval_harness.py
│   ├── test_preference_pairs.py
│   └── test_abstention.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

> **Python ≥ 3.11.** Tested on macOS arm64 (Python 3.13) and Linux x86_64 (Python 3.11, CUDA 12.2).

```bash
# 1. clone
git clone https://github.com/Mriyazat/CoT_DDI.git
cd CoT_DDI

# 2. virtual env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. spaCy model for taxonomy parsing
python -m spacy download en_core_web_sm

# 5. make the repo importable as a package
export PYTHONPATH="$(pwd)"
```

> **GPU / Cluster note.** `torch-geometric` and `vLLM` are not pinned here — install the CUDA-matched build separately on your training node. `rdkit` is optional (only needed for the MOR retrieval-gate audit); installation differs on HPC systems that ship a dummy wheel.

### Required data

The pipeline starts from a licensed **DrugBank XML release** (we used `drugbank_2026-04.xml`, SHA-256 verified in `configs/base.yaml`). DrugBank is not redistributable — obtain it from <https://go.drugbank.com/releases> and place it at:

```
data_raw/drugbank_2026-04.xml
```

Optional auxiliary sources used by the audits (paths set in `configs/base.yaml`):

- **DDInter 2** (severity metadata, never used as a label) → `data_raw/ddinter_2/`
- **KEGG / SMPDB** pathway dumps → `data_raw/pathways/`

---

## End-to-End Pipeline

The pipeline has **four phases (A → D)**. Each step writes parquet/JSONL artefacts that downstream steps read. Run them in order; every script is a self-contained `python -m` entry point.

### Phase A — Data, Taxonomy, Splits

```bash
# A1  Parse DrugBank XML → parquet (drugs, pairs, pathways, x-refs, brands, …)
python -m src.data.parse_drugbank

# A2  Pathway / target enrichment (KEGG + SMPDB)
python -m src.data.fetch_pathways
python -m src.data.build_pk_table
python -m src.data.build_signatures

# A3  Build the 7-family × 100-subtype taxonomy from DrugBank descriptions
python -m src.data.build_taxonomy

# A4  Three split protocols + 25k-pair balanced subset
python -m src.data.build_splits

# A5  Audits + GO/NO-GO freeze
python -m src.audit.a06_label_cooccurrence
python -m src.audit.a07_ddinter_severity
python -m src.audit.drug_completeness
python -m src.audit.freeze_phase_a
```

### Phase B — Multi-Teacher Consensus

Generate 24 candidate traces per pair from **each of three frontier teachers** (Llama-3.3-70B, Qwen-2.5-72B, DeepSeek-R1-Distill-70B), rule-QC them, PRM-score them, and merge into one consensus trace per pair.

```bash
# B0  Train the DDI-PRM critic (step-level rubric in configs/prm_rubric.yaml)
python -m src.teacher.prm_data
python -m src.teacher.prm_train
python -m src.teacher.prm_verify

# B1  Teacher generation  (vLLM server expected at $OPENAI_API_BASE)
python -m src.teacher.generate \
    --split subset25k \
    --teacher llama-3.3-70b \
    --candidates 24

# B2  10-gate rule QC
python -m src.teacher.qc

# B3  PRM step-level critique + rerank
python -m src.teacher.critic
python -m src.teacher.critic_rerank

# B4  Cross-teacher consensus merge + reasoning-safety filter
python -m src.teacher.merge_consensus
python -m src.teacher.apply_reasoning_safety
python -m src.teacher.audit_teacher_clean

# B5  Build preference corpora for Phase C DPO
python -m src.teacher.build_preference_pairs
python -m src.teacher.build_direction_mirror_preferences
python -m src.teacher.build_phase4_hard_negative_preferences
```

Optional: a frontier LLM-as-judge pass (GPT-5 / Claude / Gemini) for the held-out reasoning eval:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
python -m src.teacher.sample_for_judge
python -m src.teacher.llm_judge
```

### Phase C — Student Training

The student is **Qwen-2.5-7B** with LoRA (`r=64, α=128`, dropout `0.05`).

```bash
# C0  Prepare tier-weighted SFT + preference corpora
python -m src.data.prepare_phase_c
python -m src.data.build_mirror_sft_corpus

# C1  SFT — tier-weighted CE + faithfulness loss + symmetry-KL on direction tag
accelerate launch --config_file configs/accelerate_fsdp_qwen.yaml \
    -m src.training.sft_train

# C2  PRM-weighted DPO / IPO with programmatic hard-negatives
accelerate launch --config_file configs/accelerate_fsdp_qwen.yaml \
    -m src.training.dpo_mirror

# C3  (optional) classifier head on the frozen reasoner
python -m src.training.train_classifier_head
```

### Phase D — Evaluation

```bash
# D1  Build evaluation prompts (with mechanism-aware top-K neighbour block)
python -m src.data.build_student_eval_prompts

# D2  Student inference + JSON parsing + abstention
python -m src.inference.predict \
    --split pair_cold \
    --checkpoint <path-to-trained-LoRA>
python -m src.inference.abstention

# D3  XGBoost baseline (fingerprint + pharmacology features)
python -m src.evaluation.baseline_xgboost

# D4  Full 8-metric evaluation on all three splits
python -m src.evaluation.run_full_eval

# D5  Stress tests
python -m src.data.build_adversarial
python -m src.data.build_counterfactual
python -m src.data.build_polypharmacy
```

---

## Configuration

Everything is driven by `configs/base.yaml`. The most relevant knobs:

| Section            | Key                                  | Purpose                                                  |
|--------------------|--------------------------------------|----------------------------------------------------------|
| `project.seed`     | `42`                                 | global RNG seed                                          |
| `paths.*`          | `data_raw`, `data_processed`, …      | I/O roots (relative to repo)                             |
| `data_sources`     | `drugbank.sha256`                    | gate that aborts the pipeline if the XML is tampered     |
| `taxonomy`         | `max_families`, `max_subtypes`       | 7 × 100 hierarchical label space                         |
| `splits`           | `random_full`, `drug_cold`, `pair_cold`, `subset25k` | leakage-safe split sizes                |
| `models.student`   | `hf_id`, `lora_*`                    | base student model + LoRA config                         |
| `models.teacher`   | `candidates_per_pair=5..24`          | teacher decoding & sampling                              |
| `training.sft`     | `faithfulness_loss_weight`, `symmetry_loss_weight` | aux loss weights                       |
| `training.dpo`     | `beta`, `prm_weight_exponent`, `mirror_pair_ratio` | preference-loss shaping                |
| `abstention`       | `target_coverage=0.90`               | conformal + entropy gating                               |
| `metrics.go_no_go` | thresholds                           | abort the pipeline if SFT collapses (`MFS < 0.60`, …)    |

Per-phase overrides live in the `accelerate_fsdp*.yaml` files (FSDP sharding strategy, gradient checkpointing, mixed precision).

---

## Metrics

The eval harness (`src/evaluation/run_full_eval.py`) reports **8 metrics**, each implemented in its own module under `src/metrics/` so they can be unit-tested in isolation:

| Metric  | File           | What it measures                                                       |
|---------|----------------|------------------------------------------------------------------------|
| Macro-F1| —              | label classification (the standard DDI metric)                         |
| MFS     | `mfs.py`       | **Mirror-Flip Score** — agreement under AB ↔ BA swap                   |
| MPS     | `mps.py`       | **Mechanism-Path Score** — does the trace cite the right enzyme/path?  |
| CSA     | `csa.py`       | **Counterfactual Sensitivity Accuracy** — flips PK flag → flips label  |
| RPC     | `rpc.py`       | **Rare-class / Polypharmacy Coverage**                                 |
| AU      | `au.py`        | **Abstention Utility** at 90 % coverage                                |
| HR      | `hr.py`        | **Hallucination Rate** (entities not in evidence pool)                 |
| THS     | `ths.py`       | **Trace Halt Score** — stops at the right reasoning depth              |

Auxiliary metrics: `cfs` (Consensus-Family Score), `slfs` (Step-Level Faithfulness), `mor` (Mechanism-Of-Action Retrieval gate), `ris` (Retrieval-Influence Score).

---

## Testing

```bash
pytest -q
```

Covers metric correctness, the preference-pair builder, the abstention calibrator, and the eval harness end-to-end on a tiny fixture.

---

## Citation

If you use this code or the released checkpoints / corpus, please cite:

```bibtex
@inproceedings{cot_ddi_2026,
  title  = {Chain-of-Thought Distillation for Drug--Drug Interaction Prediction},
  author = {Mriyazat},
  year   = {2026},
  note   = {Code: https://github.com/Mriyazat/CoT_DDI}
}
```

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE).

DrugBank, DDInter, KEGG, and SMPDB are **not** included in this repository; obtain them from their respective providers under their own terms.
