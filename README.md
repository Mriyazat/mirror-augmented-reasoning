<div align="center">

# CoT&#8209;DDI

### Mirror-Augmented Reasoning Distillation with PRM-Weighted Preference Optimization for Drug–Drug Interaction Mechanism Prediction

<br>

<p>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.13-3776AB?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="Transformers" src="https://img.shields.io/badge/🤗_Transformers-4.42-yellow"></a>
  <a href="#"><img alt="TRL" src="https://img.shields.io/badge/🤗_TRL-0.9.4-blueviolet"></a>
  <a href="#"><img alt="vLLM" src="https://img.shields.io/badge/vLLM-0.16-1e90ff"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-22c55e"></a>
  <a href="#"><img alt="Lint" src="https://img.shields.io/badge/lint-ruff-orange"></a>
</p>

<sub>A 7B reasoner that explains the <em>mechanism</em>, not just the label — and stays consistent when you flip the pair.</sub>

</div>

<br>

> **TL;DR** &nbsp; Three frontier teachers each generate many candidate traces per drug pair under a temperature schedule. A fine-tuned **DDI-PRM critic** plus a stack of deterministic rule gates collapse those candidates into a single consensus trace. A 7B Qwen student is then trained in three stages — **SFT with a position-restricted symmetry-KL loss** (mirror AB↔BA agreement), **PRM-weighted DPO**, and a hard-negative polish — and evaluated under an 8-metric suite on three split protocols of increasing generalisation difficulty.

<br>

## Pipeline

<table align="center">
<tr>
  <td align="center" width="22%">
    <sub><b>PHASE A</b></sub><br>
    <b>Corpus &amp; Retrieval</b><br><br>
    <sub>DrugBank parse · hierarchical taxonomy · leakage-safe splits · 4-component mechanism-aware retrieval index</sub>
  </td>
  <td align="center" width="3%"><sub>→</sub></td>
  <td align="center" width="22%">
    <sub><b>PHASE B</b></sub><br>
    <b>Multi-Teacher Consensus</b><br><br>
    <sub>3 frontier teachers × many candidates · rule QC gates · DDI-PRM critic · cross-LLM merge</sub>
  </td>
  <td align="center" width="3%"><sub>→</sub></td>
  <td align="center" width="22%">
    <sub><b>PHASE C</b></sub><br>
    <b>Student Training</b><br><br>
    <sub>Qwen-2.5-7B + LoRA · tier-weighted SFT + symmetry-KL · PRM-weighted DPO · hard-negative polish</sub>
  </td>
  <td align="center" width="3%"><sub>→</sub></td>
  <td align="center" width="22%">
    <sub><b>PHASE D</b></sub><br>
    <b>Evaluation</b><br><br>
    <sub>JSON-constrained inference · conformal abstention · 8-metric suite on 3 splits</sub>
  </td>
</tr>
</table>

```mermaid
flowchart LR
    classDef phase fill:#0f172a,stroke:#475569,stroke-width:1px,color:#e2e8f0,rx:8,ry:8
    classDef ground fill:#1e293b,stroke:#334155,stroke-width:1px,color:#cbd5e1,rx:6,ry:6

    A1["DrugBank XML"]:::ground
    A2["Taxonomy<br/>(families · subtypes · direction)"]:::ground
    A3["Splits<br/>random_full · drug_cold · pair_cold"]:::ground
    A4["Retrieval index<br/>pathway · protein · ATC · SMILES"]:::ground

    B1["Teacher generation<br/>vLLM · temperature schedule"]:::ground
    B2["Rule QC gates<br/>G1 … G10"]:::ground
    B3["DDI-PRM step critic"]:::ground
    B4["Cross-LLM consensus<br/>+ reasoning-safety filter"]:::ground

    C1["SFT<br/>tier-weighted CE<br/>+ faithfulness<br/>+ symmetry-KL"]:::ground
    C2["PRM-weighted DPO<br/>exact hook · IS fallback"]:::ground
    C3["Hard-negative polish<br/>family · axis · subtype · direction"]:::ground

    D1["JSON-constrained<br/>inference"]:::ground
    D2["Conformal +<br/>entropy abstention"]:::ground
    D3["8-metric suite<br/>× 3 split protocols"]:::ground

    A1 --> A2 --> A3 --> A4 --> B1
    B1 --> B2 --> B3 --> B4 --> C1
    C1 --> C2 --> C3 --> D1
    D1 --> D2 --> D3
```

<br>

## Why this exists

Most DDI benchmarks score **one number** — the top-1 label of a feature or graph classifier. In the clinic that is not enough. A pharmacist needs the **mechanism** (which CYP, transporter, or pharmacodynamic axis), the **direction** (A→B, B→A, bidirectional, or none), the **evidence**, and a calibrated **abstention** when the evidence is thin.

Three failure modes block naive teacher-to-student distillation:

<table>
<thead>
<tr><th align="left" width="22%">Failure</th><th align="left" width="42%">What it looks like</th><th align="left" width="36%">How we address it</th></tr>
</thead>
<tbody>
<tr>
  <td><b>Mirror inconsistency</b></td>
  <td>The same pair flipped AB↔BA gets different family / direction predictions.</td>
  <td>Co-batched AB &amp; BA records + position-restricted symmetry-KL on the direction tag.</td>
</tr>
<tr>
  <td><b>Class imbalance</b></td>
  <td>One family dominates; rare families are abstained away under naive cross-entropy.</td>
  <td>Class-balanced <code>1/√n<sub>f</sub></code> sampling + family-axis hard-negatives.</td>
</tr>
<tr>
  <td><b>Reasoning decay</b></td>
  <td>Student parrots teacher phrasing and cites phantom evidence.</td>
  <td>DDI-PRM step critic + 10 rule QC gates + reasoning-safety filter.</td>
</tr>
</tbody>
</table>

<br>

## Repository layout

```
CoT_DDI/
├── configs/                       YAML configs for every phase
│   ├── base.yaml                  paths · splits · models · loss weights · GO/NO-GO thresholds
│   ├── prm_rubric.yaml            step-level rubric for the DDI-PRM
│   └── accelerate_fsdp*.yaml      FSDP / mixed-precision per training stage
│
├── scripts/                       Pipeline runners (one shell wrapper per phase)
│   ├── download_models.py               HF snapshot of every teacher / PRM / student
│   ├── run_phase_a.sh                   Phase A — corpus / taxonomy / splits / audits
│   ├── run_phase_b.sh                   Phase B — PRM · teachers · QC · consensus · prefs
│   ├── run_phase_c.sh                   Phase C — SFT + symmetry-KL · PRM-DPO · head
│   ├── run_phase_d.sh                   Phase D — inference · abstention · eval · stress
│   └── run_all.sh                       end-to-end A → D
│
├── src/
│   ├── data/                      Phase A — corpus construction
│   │   ├── parse_drugbank.py            XML → parquet (drugs, pairs, pathways, x-refs, brands)
│   │   ├── fetch_pathways.py            KEGG + SMPDB pathway harvest
│   │   ├── build_pk_table.py            CYP / P-gp / OATP / BCRP flags per drug
│   │   ├── build_signatures.py          pair-level pathway- &amp; protein-Jaccard signatures
│   │   ├── build_taxonomy.py            hierarchical mechanism taxonomy
│   │   ├── build_splits.py              random_full / drug_cold / pair_cold + subset
│   │   ├── build_mirror_sft_corpus.py   AB + BA mirror records for SFT
│   │   ├── build_adversarial.py         direction-flip / negation stress set
│   │   ├── build_counterfactual.py      single-PK-flag perturbations
│   │   ├── build_polypharmacy.py        3-drug combinations
│   │   ├── build_student_eval_prompts.py prompts with top-K retrieval block
│   │   └── prepare_phase_c.py           tier-weighted SFT + preference corpus
│   │
│   ├── audit/                     Phase A audits + GO/NO-GO freeze
│   ├── teacher/                   Phase B — generation, QC, PRM, consensus
│   │   ├── prompt.py · schema.py · context_builder.py · provider.py
│   │   ├── generate.py                  vLLM-backed candidate generation per teacher
│   │   ├── qc.py                        10 rule gates G1 … G10
│   │   ├── critic.py · prm_data.py · prm_train.py · prm_verify.py
│   │   ├── critic_rerank.py             PRM rerank within teacher
│   │   ├── merge.py · merge_consensus.py cross-LLM consensus merge
│   │   ├── apply_reasoning_safety.py    citation / direction-verb filter
│   │   ├── llm_judge.py                 GPT / Claude / Gemini OOF probe
│   │   ├── build_preference_pairs.py
│   │   ├── build_direction_mirror_preferences.py
│   │   └── build_phase4_hard_negative_preferences.py
│   │
│   ├── training/                  Phase C — student
│   │   ├── sft_train.py                 tier-weighted SFT + faithfulness + symmetry-KL
│   │   ├── dpo_mirror.py                PRM-weighted DPO / IPO (exact hook + IS fallback)
│   │   ├── train_classifier_head.py
│   │   └── evaluate_sweep.py · summarize_sweep.py
│   │
│   ├── inference/                 Phase D — prediction
│   │   ├── predict.py                   JSON-constrained inference with retrieval block
│   │   ├── abstention.py                conformal + entropy gating
│   │   └── augment_predictions.py
│   │
│   ├── evaluation/                Phase D — eval harness + baselines
│   │   ├── run_full_eval.py             full 8-metric suite on all 3 splits
│   │   └── baseline_xgboost.py          gradient-boosted reference over the same features
│   │
│   └── metrics/                   one module per metric, unit-tested
│       mfs.py  mps.py  csa.py  rpc.py  au.py  hr.py  ths.py
│       cfs.py  slfs.py  mor.py  ris.py
│
├── tests/                         pytest unit tests
└── requirements.txt
```

<br>

## Installation

> **Python ≥ 3.11.** Verified on macOS arm64 and Linux x86_64 with CUDA 12.2.

```bash
git clone https://github.com/Mriyazat/CoT_DDI.git
cd CoT_DDI

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

export PYTHONPATH="$(pwd)"
```

> **GPU note.** `torch-geometric` and `vLLM` are not pinned in `requirements.txt` — install the CUDA-matched build separately on your training node. `rdkit` is optional (only the MOR retrieval-gate audit uses it).

### Data

DrugBank is licensed and not redistributable. Download `drugbank_2026-04.xml` from <https://go.drugbank.com/releases> and place it at:

```
data_raw/drugbank_2026-04.xml
```

Optional auxiliary sources (paths in `configs/base.yaml`):

- **DDInter 2** &nbsp;— severity metadata, *never* used as a label &nbsp;→ &nbsp;`data_raw/ddinter_2/`
- **KEGG / SMPDB** &nbsp;— pathway dumps &nbsp;→ &nbsp;`data_raw/pathways/`

<br>

## Run the pipeline

Every step is a self-contained `python -m` entry point that reads/writes parquet & JSONL artefacts. Run them in order — either via the per-phase shell wrappers under `scripts/`:

```bash
# end-to-end
bash scripts/run_all.sh

# or one phase at a time
bash scripts/run_phase_a.sh
bash scripts/run_phase_b.sh
bash scripts/run_phase_c.sh
CHECKPOINT=outputs/student/dpo/<run>  bash scripts/run_phase_d.sh
```

…or by invoking the underlying entry points directly:

### A · Corpus, taxonomy, splits

```bash
# A1 — parse DrugBank XML → parquet (drugs, pairs, pathways, x-refs, brands)
python -m src.data.parse_drugbank

# A2 — pathway / target enrichment (KEGG + SMPDB) and per-drug PK flags
python -m src.data.fetch_pathways
python -m src.data.build_pk_table
python -m src.data.build_signatures

# A3 — hierarchical mechanism taxonomy
python -m src.data.build_taxonomy

# A4 — three split protocols + balanced teacher subset
python -m src.data.build_splits

# A5 — audits + GO/NO-GO freeze
python -m src.audit.a06_label_cooccurrence
python -m src.audit.a07_ddinter_severity
python -m src.audit.drug_completeness
python -m src.audit.freeze_phase_a
```

### B · Multi-teacher consensus

```bash
# B0 — train the DDI-PRM critic (rubric: configs/prm_rubric.yaml)
python -m src.teacher.prm_data
python -m src.teacher.prm_train
python -m src.teacher.prm_verify

# B1 — candidate generation per teacher under a temperature schedule
#       (vLLM server expected at $OPENAI_API_BASE)
python -m src.teacher.generate --split subset --teacher llama-3.3-70b
python -m src.teacher.generate --split subset --teacher qwen-2.5-72b
python -m src.teacher.generate --split subset --teacher deepseek-r1-70b

# B2 — 10 deterministic rule gates G1 … G10
python -m src.teacher.qc

# B3 — PRM step-level critic + best-of rerank within each teacher
python -m src.teacher.critic
python -m src.teacher.critic_rerank

# B4 — cross-LLM consensus merge + reasoning-safety filter
python -m src.teacher.merge_consensus
python -m src.teacher.apply_reasoning_safety
python -m src.teacher.audit_teacher_clean

# B5 — preference corpora for Phase C
python -m src.teacher.build_preference_pairs
python -m src.teacher.build_direction_mirror_preferences
python -m src.teacher.build_phase4_hard_negative_preferences
```

<details>
<summary><b>Optional · frontier LLM-as-judge OOF probe (GPT / Claude / Gemini)</b></summary>

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
python -m src.teacher.sample_for_judge
python -m src.teacher.llm_judge
```

</details>

### C · Student training (Qwen-2.5-7B + LoRA)

```bash
# C0 — tier-weighted SFT + mirror preference corpora
python -m src.data.prepare_phase_c
python -m src.data.build_mirror_sft_corpus

# C1 — SFT with tier-weighted CE + faithfulness + symmetry-KL on the direction tag
accelerate launch --config_file configs/accelerate_fsdp_qwen.yaml \
    -m src.training.sft_train

# C2 — PRM-weighted DPO / IPO with four programmatic hard-negative families
accelerate launch --config_file configs/accelerate_fsdp_qwen.yaml \
    -m src.training.dpo_mirror

# C3 — (optional) classifier head over the frozen reasoner
python -m src.training.train_classifier_head
```

### D · Evaluation

```bash
# D1 — build evaluation prompts with the top-K mechanism-aware neighbour block
python -m src.data.build_student_eval_prompts

# D2 — student inference + JSON-constrained parse + abstention
python -m src.inference.predict     --split pair_cold --checkpoint <path-to-LoRA>
python -m src.inference.abstention

# D3 — gradient-boosted reference over the same 4-component features
python -m src.evaluation.baseline_xgboost

# D4 — full 8-metric suite on random_full / drug_cold / pair_cold
python -m src.evaluation.run_full_eval

# D5 — stress sets
python -m src.data.build_adversarial
python -m src.data.build_counterfactual
python -m src.data.build_polypharmacy
```

<br>

## Method at a glance

### 1 · Position-restricted symmetry-KL — the mirror constraint

For every co-batched (AB, BA) pair the loss is

$$
\mathcal{L}\bigl(p_{\text{AB}}, p_{\text{BA}}\bigr) \;=\; \mathcal{L}_{\text{SFT}}(p_{\text{AB}}) + \mathcal{L}_{\text{SFT}}(p_{\text{BA}}) \;+\; \lambda \cdot \mathrm{KL}\!\left( \mathrm{softmax}(z^{\text{AB}}_{\text{tag}}) \;\big\|\; T_{\pi}\bigl[\mathrm{softmax}(z^{\text{BA}}_{\text{tag}})\bigr] \right)
$$

where $T_{\pi}$ permutes the four direction tokens (AB↔BA, BIDIR↔BIDIR, N/A↔N/A). The KL fires **only** on the direction-tag token, leaving the free-form reasoning uncoupled — which is the key reason it works.

### 2 · PRM-weighted DPO

Per-pair weight from the DDI-PRM margin between chosen and rejected:

$$
w_i \;=\; \mathrm{clip}\!\bigl(\Phi_{\text{PRM}}(y^+_i) - \Phi_{\text{PRM}}(y^-_i),\; 0,\; 1\bigr), \qquad \mathcal{L}_{\text{PRM-DPO}} \;=\; -\sum_i w_i \log \sigma\bigl(\beta\,\Delta_i\bigr)
$$

Two interchangeable backends, dispatched by runtime capability detection:

<table>
<thead>
<tr><th align="left">Backend</th><th align="left">When it is used</th><th align="left">How it implements Eq. (above)</th></tr>
</thead>
<tbody>
<tr><td><b>Exact</b></td><td>TRL exposes a per-example <code>dpo_loss</code> hook</td><td>Monkey-patch the hook; multiply per-example losses by <code>w<sub>i</sub></code> before reduction.</td></tr>
<tr><td><b>IS fallback</b></td><td>older TRL without the hook</td><td>Deterministic importance sampling: minibatches drawn ∝ <code>w<sub>i</sub></code>, standard DPO loss applied.</td></tr>
</tbody>
</table>

### 3 · Four hard-negative families

All edits are confined to the `final_answer` block so the trace prefix is identical to the chosen — preventing the student from exploiting surface artefacts.

<table>
<thead>
<tr><th align="left" width="34%">Family</th><th align="left" width="36%">Construction</th><th align="left" width="30%">Targets</th></tr>
</thead>
<tbody>
<tr><td><code>FAMILY-SWAP-TO-ADVERSERISK</code></td><td>rewrite final family to the dominant family</td><td>over-prediction attractor</td></tr>
<tr><td><code>FAMILY-AXIS SWAP</code></td><td>swap family across a curated confusion-axis map</td><td>cross-family confusions</td></tr>
<tr><td><code>SUBTYPE SWAP</code></td><td>keep family, sample a different in-family subtype</td><td>sub-family confusion</td></tr>
<tr><td><code>DIRECTION FLIP</code></td><td>apply <code>T<sub>π</sub></code> to flip the direction tag</td><td>direction errors</td></tr>
</tbody>
</table>

### 4 · Mechanism-aware retrieval

Drug–drug similarity is a 4-component score:

$$
s(d_i, d_j) \;=\; w_p\, J_p \;+\; w_r\, J_r \;+\; w_a\, \tfrac{A}{7} \;+\; w_t\, T
$$

with $J_p$ = pathway Jaccard (SMPDB ∪ KEGG), $J_r$ = protein-target Jaccard, $A$ = deepest common ATC prefix depth, and $T$ = SMILES Tanimoto over Morgan-2 fingerprints. Pair–pair score takes the max of the two alignment options; the top-$K$ neighbour universe is restricted to the training split so test-side drugs never leak.

<br>

## Configuration

Everything is driven by `configs/base.yaml`. The few knobs you usually touch:

```yaml
project:
  seed: 42

models:
  student:
    hf_id: Qwen/Qwen2.5-7B
    lora_r: 64
    lora_alpha: 128

  teacher:
    candidates_per_pair: 24
    decoding:
      temperature: 0.7
      top_p: 0.9
      max_tokens: 1024

training:
  sft:
    epochs: 3
    lr: 2.0e-4
    faithfulness_loss_weight: 0.5
    symmetry_loss_weight: 0.3
  dpo:
    beta: 0.1
    prm_weight_exponent: 1.0
    mirror_pair_ratio: 0.5

abstention:
  method: conformal_plus_entropy
  target_coverage: 0.90

retrieval:
  top_k: 8
  mor_floor: 0.55       # MOR validation gate
```

Per-phase FSDP / mixed-precision settings live in the `configs/accelerate_fsdp*.yaml` files.

<br>

## Metrics

The eval harness emits **eight metrics**, each in its own unit-tested module under `src/metrics/`.

<table>
<thead>
<tr><th align="left" width="10%">#</th><th align="left" width="22%">Metric</th><th align="left">Measures</th></tr>
</thead>
<tbody>
<tr><td align="center">1</td><td>macro-F1</td><td>Family classification — the standard DDI metric.</td></tr>
<tr><td align="center">2</td><td><b>MFS</b> &nbsp; <code>mfs.py</code></td><td>Mirror Family Stability — fraction of pairs whose AB and BA predictions agree on family.</td></tr>
<tr><td align="center">3</td><td><b>MPS</b> &nbsp; <code>mps.py</code></td><td>Mirror Prediction Symmetry — full (family, subtype, direction) triple agrees after applying T<sub>π</sub>.</td></tr>
<tr><td align="center">4</td><td><b>CSA</b> &nbsp; <code>csa.py</code></td><td>Context-Support Alignment — prediction is supported by a verbatim cited identifier from the evidence pool.</td></tr>
<tr><td align="center">5</td><td><b>RPC</b> &nbsp; <code>rpc.py</code></td><td>Reasoning-Path Coherence — mean step-level PRM score.</td></tr>
<tr><td align="center">6</td><td><b>AU</b> &nbsp; <code>au.py</code></td><td>Abstention Utility — area under coverage-vs-accuracy curve under conformal abstention.</td></tr>
<tr><td align="center">7</td><td><b>HR</b> &nbsp; <code>hr.py</code></td><td>Hallucination Rate — predictions citing identifiers not in the evidence pool.</td></tr>
<tr><td align="center">8</td><td><b>THS</b> &nbsp; <code>ths.py</code></td><td>Tiered-Hierarchy Score — credit only when family, subtype <em>and</em> direction are all correct.</td></tr>
</tbody>
</table>

Auxiliary metrics: <code>cfs</code> (consensus-family score), <code>slfs</code> (step-level faithfulness), <code>mor</code> (mechanism-of-action retrieval gate), <code>ris</code> (retrieval-influence score).

<br>

## Tests

```bash
pytest -q
```

Covers every metric (`test_metrics.py`), the eval harness end-to-end on a tiny fixture (`test_eval_harness.py`), preference-pair construction (`test_preference_pairs.py`), and the abstention calibrator (`test_abstention.py`).

<br>

## License

[MIT](LICENSE).

DrugBank, DDInter, KEGG, and SMPDB are **not** included in this repository — obtain them from their respective providers under their own terms.
