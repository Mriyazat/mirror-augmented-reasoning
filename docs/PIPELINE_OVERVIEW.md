# Pipeline Overview

A file-level tour of the four pipeline phases. Commands and environment
setup are documented in the top-level `README.md`; student-training
hyperparameter defaults are listed in `docs/PHASE_C_HYPERPARAMS.md`.

## Phase A — Data construction

```
src/data/parse_drugbank.py        DrugBank XML -> drugs.parquet, pairs.parquet
src/data/build_taxonomy.py        7-family / multi-subtype label taxonomy
src/data/build_pk_table.py        CYP / P-gp / OATP flags per drug
src/data/build_signatures.py      Pair-level mechanistic signatures
src/data/fetch_pathways.py        KEGG + SMPDB pathway graph
src/data/build_splits.py          random_full / drug_cold / pair_cold splits
src/data/build_stratified_manifest.py
                                  25k stratified development subset
src/data/build_polypharmacy.py    Polypharmacy (3-drug) evaluation set
src/data/build_adversarial.py     Adversarial retrieval probes
src/data/build_counterfactual.py  PK-flag counterfactual probe set
src/audit/                        Integrity audits and freeze script
```

## Phase B — Teacher generation and PRM

```
src/teacher/prompt.py             User-prompt template (with neighbors)
src/teacher/context_builder.py    Pathway / target / PK / neighbor context
src/teacher/schema.py             Step schema + final-answer schema
src/teacher/generate.py           vLLM-served candidate generation (resumable)
src/teacher/qc.py                 Hedging, abstention, summary gates
src/teacher/critic.py             Programmatic per-step verifier
src/teacher/prm_data.py           Build PRM step-level training file
src/teacher/prm_train.py          PRM fine-tune (Med-PRM compatible)
src/teacher/prm_verify.py         PRM evaluation
src/teacher/score_traces_prm.py   PRM-score every candidate trace
src/teacher/critic_rerank.py      PRM-guided refine / keep / drop
src/teacher/merge.py              Per-pair winner merge across teachers
src/teacher/merge_consensus.py    Cross-teacher consensus tier
src/teacher/build_preference_pairs.py
                                  Hard-negative + mirror preference pairs
src/teacher/llm_judge.py          Out-of-family LLM-as-judge QC
src/teacher/apply_reasoning_safety.py
                                  Strip mid-chain "silent abstain" language
```

## Phase C — Student SFT and Mirror DPO

```
src/data/build_mirror_sft_corpus.py
                                  AB+BA mirror augmentation of SFT corpus
src/data/prepare_phase_c.py       Train/val/smoke splits + tier weights
src/training/sft_train.py         LoRA SFT with faithfulness + symmetry-KL
src/training/dpo_mirror.py        Mirror-IPO / DPO on hard-negative pairs
src/training/build_trace_alignment_sft.py
                                  Trace-align SFT (rescue + demo records)
src/training/train_classifier_head.py
                                  Optional classification-head head
src/training/evaluate_sweep.py    Sweep evaluation
src/training/summarize_sweep.py   Sweep summary tables
```

## Phase D — Evaluation

```
src/inference/predict.py                       Vanilla student inference
src/inference/predict_with_rerank.py           Multi-decode + PRM re-rank
src/inference/predict_two_stage.py             Family-first then subtype
src/inference/predict_with_hf_local.py         Local HF generation path
src/inference/predict_with_frontier_llm.py     OpenAI / Anthropic / Google
src/inference/predict_with_frontier_llm_full_traces.py
                                               Closed-weight + full trace
src/inference/aggregate_rerank.py              Late-fusion rerank
src/inference/self_consistency.py              Majority over decodes
src/inference/frontier_rescue.py               Frontier-LLM rescue path
src/inference/abstention.py                    Conformal + entropy abstention
src/inference/calibrate_conformal.py           Per-family conformal calibration
src/inference/apply_cpu_stack.py               CPU post-hoc stack
src/inference/augment_predictions.py           Attach gold labels for eval
src/inference/trace_rescue.py                  Trace-majority rescue heuristic

src/evaluation/run_full_eval.py                Headline harness
src/evaluation/headline_metrics.py             Macro-F1 / subtype-acc / direction
src/evaluation/bootstrap_ci.py                 95 % bootstrap CIs
src/evaluation/paired_bootstrap_significance.py
                                               Paired significance tests
src/evaluation/per_class_bootstrap.py          Per-class CIs
src/evaluation/conformal_recalibrate.py        Re-fit conformal threshold
src/evaluation/honest_router_eval.py           Verifier-as-router evaluation
src/evaluation/ensemble_eval.py                Ensemble of decodes
src/evaluation/family_prior_rebalance.py       Family-prior debiasing
src/evaluation/frontier_compare.py             Side-by-side with frontier LLMs
src/evaluation/baseline_fast_suite.py          OpenDDI / GNN baselines
src/evaluation/baseline_xgboost.py             XGBoost (fingerprint + PK)
src/evaluation/llm_vs_llm.py                   Judge-on-judge consistency
src/evaluation/leakage_probe.py                Train / test leakage probe
src/evaluation/anti_memorisation_check.py      Mnemonic-prompt probe
```

## Metrics library

```
src/metrics/cfs.py    Counterfactual-Faithfulness Score
src/metrics/csa.py    Compositional-Subtype Accuracy
src/metrics/hr.py     Hallucination Rate
src/metrics/mfs.py    Mechanism-Faithfulness Score
src/metrics/mor.py    Mechanism-Overlap Ratio (RDKit optional)
src/metrics/mps.py    Mirror-Pair Separation
src/metrics/rpc.py    Reasoning-Prediction Coherence
src/metrics/slfs.py   Selective-Label Faithfulness Score
src/metrics/ths.py    Taxonomy-Hierarchy Score
src/metrics/au.py     Abstention Utility (selective coverage)
```
