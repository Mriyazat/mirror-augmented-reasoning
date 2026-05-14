"""DDI teacher trace generation + PRM (Phase B).

Modules:
    schema          — dataclasses + rubric loader + step/trace validators
    context_builder — B1a: per-pair retrieval context from Phase-A artifacts
    prompt          — B1b: teacher prompt builder from rubric + context
    provider        — B1b: provider-agnostic LLM client (vLLM / Ollama / OpenAI)
    generate        — B1b: run teacher, emit raw candidate traces
    qc              — B2: 5-gate QC filter with auto-derived gold labels
    prm_data        — B3: build Med-PRM-format PRM training data from QC'd traces
    critic          — B4: PRM-guided best-of-N + refinement
    merge           — B5: dedupe + stratify + write teacher_clean.jsonl
"""
