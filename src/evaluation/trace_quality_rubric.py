"""EMNLP-grade trace quality rubric for LLM-as-judge evaluation.

DESIGN PRINCIPLES (paper §X.Y methodology):
  1. LENGTH-INVARIANT: each dimension scored on per-claim correctness or
     per-step precision, NOT total word count. Calibration examples include
     both short (4-step) and long (12-step) high-quality traces to anchor
     the judge against verbosity bias.
  2. BLIND: model identity is stripped before prompting the judge; trace
     IDs randomized; family hints removed where possible.
  3. CROSS-JUDGE: no model judges its own outputs (self-preference bias).
     Each frontier LLM (Claude, GPT-4o, Gemini) is judged ONLY by the
     other two; the distilled student is judged by all three; the per-
     dimension scores are bootstrapped over judges to estimate uncertainty.
  4. CITED EVIDENCE: rubric requires the judge to QUOTE specific text
     spans from the trace as justification for each dimension score, so
     scores can be audited post-hoc.

See `JUDGE_SYSTEM_PROMPT` for the actual instructions sent to each judge.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# RUBRIC (the 6 dimensions)
# --------------------------------------------------------------------------
RUBRIC_DIMENSIONS = [
    {
        "id": "factuality",
        "name": "Factual correctness",
        "max_score": 8,
        "definition": (
            "Are the biomedical claims in the trace factually correct? "
            "Examples: does drug X actually inhibit CYP3A4? does this drug "
            "actually target this receptor? IGNORE LENGTH: 3 correct claims "
            "score the same as 5 correct claims if both fully cover the "
            "essential mechanism."
        ),
        "anchors": {
            8: "All claims are factually correct (no errors); a domain expert would agree with every assertion.",
            6: "Mostly correct (>=80% of claims correct); minor inaccuracies that do not affect the conclusion.",
            4: "Mixed (60-80% correct); some material errors that weaken the reasoning.",
            2: "Mostly incorrect (40-60% correct); the trace gets fundamental biology wrong.",
            0: "Almost entirely incorrect or no verifiable claims at all.",
        },
    },
    {
        "id": "faithfulness",
        "name": "Faithfulness to final answer",
        "max_score": 8,
        "definition": (
            "Do the reasoning steps logically support the trace's final "
            "family/subtype/direction answer? IGNORE LENGTH: a short trace "
            "with tight logical chain scores the same as a long trace "
            "with the same logical chain. Penalize ONLY when steps "
            "contradict the final answer or jump to a conclusion not "
            "supported by the cited evidence."
        ),
        "anchors": {
            8: "Every step contributes to the final answer; the conclusion follows directly from the reasoning chain.",
            6: "Most steps support the final answer; one or two tangential or weakly-connected steps.",
            4: "Some steps support the conclusion; the chain is partially broken or jumps to the answer.",
            2: "Reasoning is largely disconnected from the final answer; conclusion appears asserted not derived.",
            0: "Steps contradict the final answer, or final answer is absent.",
        },
    },
    {
        "id": "grounding",
        "name": "Evidence grounding",
        "max_score": 8,
        "definition": (
            "When the trace cites evidence IDs (DrugBank IDs like DB12345, "
            "UniProt IDs like P12345, pathway IDs like SMP0001234, ATC "
            "codes, etc.), are they (a) drawn from the evidence pool that "
            "was provided in the query, and (b) actually relevant to the "
            "claim being supported? IGNORE LENGTH/COUNT: precision of "
            "citation matters, not count. A trace citing 3 perfectly "
            "relevant IDs scores the same as one citing 8 relevant IDs."
        ),
        "anchors": {
            8: "Every cited ID appears in the evidence pool AND is relevant to the claim it supports.",
            6: "Most citations are valid and relevant; a few weak or marginal but no fabricated IDs.",
            4: "Mixed: some citations correct, some IDs cited that do not appear in the pool, or weak/irrelevant.",
            2: "Citations are largely missing, fabricated, or unrelated to the claims.",
            0: "No usable citations OR all cited IDs are fabricated.",
        },
    },
    {
        "id": "specificity",
        "name": "Mechanism specificity",
        "max_score": 8,
        "definition": (
            "Does the trace name the SPECIFIC pharmacological mechanism "
            "(e.g., 'competitively inhibits CYP3A4 at the catalytic site') "
            "rather than vague descriptions (e.g., 'affects metabolism')? "
            "IGNORE LENGTH: a 2-sentence step that says 'X competitively "
            "inhibits CYP3A4' is fully specific. A 4-paragraph step that "
            "says 'X may affect drug metabolism in some way' is not."
        ),
        "anchors": {
            8: "Each mechanism is named precisely (specific enzyme/transporter/receptor + interaction mode).",
            6: "Most mechanisms specific; one or two general statements.",
            4: "Mixed: about half specific, half generic/hand-wavy.",
            2: "Predominantly vague ('affects', 'modulates', 'interacts with') without naming concrete targets.",
            0: "No mechanism stated, or only family-level handwaving.",
        },
    },
    {
        "id": "hallucination",
        "name": "Hallucination check (lower = worse)",
        "max_score": 8,
        "definition": (
            "Does the trace INVENT facts not supported by the evidence "
            "pool? Examples: invented drug interactions, fabricated enzyme "
            "names, made-up pathway IDs, claims about FDA warnings that "
            "aren't in the evidence. Score HIGH (8) when the trace stays "
            "strictly within what the evidence supports; score LOW (0) "
            "when it confabulates. IGNORE LENGTH: a short trace that "
            "stays grounded scores 8; a long trace with one fabrication "
            "drops to 6 or lower."
        ),
        "anchors": {
            8: "Zero fabrications; every factual claim is either trivially well-known or backed by cited evidence.",
            6: "One minor unsupported claim (e.g., a small extrapolation beyond evidence) but no fabricated identifiers.",
            4: "Two or three unsupported claims, OR one fabricated identifier (a non-existent UniProt/SMPDB ID).",
            2: "Multiple unsupported claims and/or fabricated identifiers that materially affect the reasoning.",
            0: "Pervasive hallucination; majority of claims are fabricated or unverifiable.",
        },
    },
    {
        "id": "coherence",
        "name": "Hierarchical coherence",
        "max_score": 8,
        "definition": (
            "Does the trace move LOGICALLY from drug mechanism-of-action "
            "to interaction mechanism to clinical consequence? Or does it "
            "jump around, repeat itself, or omit critical bridging steps? "
            "IGNORE LENGTH: a 3-step trace [MoA-of-A, MoA-of-B, "
            "Interaction] is fully coherent. A 12-step trace that "
            "repeats the same MoA claim 3 times is less coherent."
        ),
        "anchors": {
            8: "Logical structure: MoA(A) -> MoA(B) -> Interaction mechanism -> Clinical/pharmacological effect.",
            6: "Mostly coherent with one redundant or out-of-place step.",
            4: "Structure exists but is loose; missing one bridging step OR has 2-3 redundant repetitions.",
            2: "Disorganized: steps appear in random order, or major bridging steps are missing.",
            0: "No logical structure detectable.",
        },
    },
]

# Composite = mean of all 6 dimensions, range [0, 8].

# --------------------------------------------------------------------------
# CALIBRATION EXAMPLES (paired short/long to anchor length-invariance)
# --------------------------------------------------------------------------
CALIBRATION_TRACE_SHORT = {
    "_caption_": "SHORT, DENSE, HIGH-QUALITY TRACE (4 steps). Should score ~7/8.",
    "steps": [
        {
            "step_id": 1,
            "role": "pk_flag",
            "claim": "Rifampin is a potent inducer of CYP3A4 (evidence: DB01045 cyp3a4_ind).",
            "evidence_ids": ["DB01045", "cyp3a4_inducer"],
            "direction_tag": "n/a",
            "family_hint": "PK_Metabolism",
        },
        {
            "step_id": 2,
            "role": "pk_flag",
            "claim": "Cyclosporine is primarily metabolized by CYP3A4 (evidence: DB00091 cyp3a4_sub).",
            "evidence_ids": ["DB00091", "cyp3a4_substrate"],
            "direction_tag": "n/a",
            "family_hint": "PK_Metabolism",
        },
        {
            "step_id": 3,
            "role": "interaction",
            "claim": "CYP3A4 induction by rifampin accelerates cyclosporine clearance, lowering plasma concentrations.",
            "evidence_ids": ["DB01045", "DB00091"],
            "direction_tag": "a_to_b",
            "family_hint": "PK_Metabolism",
        },
    ],
    "final_answer": {
        "family": "PK_Metabolism",
        "subtype": "metabolism_induction",
        "direction_tag": "a_to_b",
        "polarity": "down",
        "abstain": False,
        "confidence": 0.85,
        "summary": "Rifampin induces CYP3A4 and reduces cyclosporine exposure.",
    },
}

CALIBRATION_TRACE_LONG = {
    "_caption_": "LONG, VERBOSE, HIGH-QUALITY TRACE (12 steps). Should ALSO score ~7/8 -- length does not raise the score.",
    "steps": [
        {"step_id": i, "role": "mechanism_of_action",
         "claim": f"Step {i}: extended elaboration of the same essential mechanism — rifampin induces hepatic CYP3A4 expression via PXR activation, leading to accelerated phase-I metabolism of CYP3A4 substrates including cyclosporine.",
         "evidence_ids": ["DB01045", "DB00091", "cyp3a4_inducer"],
         "direction_tag": "a_to_b",
         "family_hint": "PK_Metabolism"}
        for i in range(1, 13)
    ],
    "final_answer": {
        "family": "PK_Metabolism",
        "subtype": "metabolism_induction",
        "direction_tag": "a_to_b",
        "polarity": "down",
        "abstain": False,
        "confidence": 0.85,
        "summary": "Rifampin induces CYP3A4 and reduces cyclosporine exposure.",
    },
}


# --------------------------------------------------------------------------
# JUDGE SYSTEM PROMPT
# --------------------------------------------------------------------------
def build_judge_system_prompt() -> str:
    rubric_text = []
    for d in RUBRIC_DIMENSIONS:
        anchors_str = "\n".join(f"      {s}: {a}" for s, a in sorted(d["anchors"].items(), reverse=True))
        rubric_text.append(
            f"  [{d['id']}] {d['name']}  (range 0-{d['max_score']}, integer)\n"
            f"    Definition: {d['definition']}\n"
            f"    Anchors:\n{anchors_str}"
        )
    rubric_block = "\n\n".join(rubric_text)

    return f"""You are an expert biomedical evaluator scoring drug-drug-interaction (DDI) reasoning traces.

YOUR TASK:
You will be shown (a) the DDI QUERY (a pair of drugs and an evidence pool), and (b) a REASONING TRACE produced by some anonymous model.

You must score the trace on SIX rubric dimensions, each on an integer scale from 0 to 8. For each dimension, you must also quote a short justification (1-2 sentences) citing specific text from the trace.

==================================================================
CRITICAL: LENGTH-INVARIANCE INSTRUCTION (read carefully)
==================================================================
Traces vary in length because some models are large and produce verbose output,
while smaller distilled models produce shorter, denser traces. **LENGTH IS NOT
A QUALITY SIGNAL.** A trace with 3 correct, specific steps deserves the SAME
SCORE as a trace with 12 correct steps covering the same essential mechanism.

DO NOT reward verbosity. DO NOT penalize brevity. DO NOT use "comprehensiveness"
or "thoroughness" as a back-door for length. Score on per-claim correctness,
specificity, and logical coherence ONLY.

To calibrate: a 3-step trace that says "Drug A inhibits CYP3A4; Drug B is a
CYP3A4 substrate; therefore A elevates B's plasma concentration" is FULLY
correct, FULLY specific, and FULLY coherent — it should score near the maximum
(7-8 / 8) on every applicable dimension, even though it is short.

A 12-step trace that says the same thing in 12 different paraphrasings should
NOT score higher. If anything, redundancy should LOWER the coherence score.
==================================================================

==================================================================
RUBRIC (6 dimensions, each scored 0-8 as INTEGER)
==================================================================
{rubric_block}
==================================================================

OUTPUT FORMAT:
You must output VALID JSON only. No prose before or after. No markdown fences.
Schema:

{{
  "scores": {{
    "factuality":     <int 0-8>,
    "faithfulness":   <int 0-8>,
    "grounding":      <int 0-8>,
    "specificity":    <int 0-8>,
    "hallucination":  <int 0-8>,
    "coherence":      <int 0-8>
  }},
  "justifications": {{
    "factuality":     "<1-2 sentences quoting specific trace text>",
    "faithfulness":   "<1-2 sentences>",
    "grounding":      "<1-2 sentences>",
    "specificity":    "<1-2 sentences>",
    "hallucination":  "<1-2 sentences>",
    "coherence":      "<1-2 sentences>"
  }},
  "length_bias_self_check": "<one sentence: 'Did I penalize this trace for being short or reward it for being long? If yes, revise above scores.'>"
}}

Return JSON ONLY. No other text.
"""


def build_judge_user_prompt(query_snippet: str, trace_text: str) -> str:
    return f"""DDI QUERY (system context + user question for the model being judged):
{query_snippet}

REASONING TRACE TO EVALUATE (model identity withheld):
{trace_text}

Score this trace on the 6 rubric dimensions. Return JSON only."""


if __name__ == "__main__":
    print(build_judge_system_prompt())
    print("\n\n--- CALIBRATION EXAMPLES ---")
    import json as _json
    print("SHORT (4 steps, target ~7/8):")
    print(_json.dumps(CALIBRATION_TRACE_SHORT, indent=2))
    print("\nLONG (12 steps, target ~7/8):")
    print(_json.dumps(CALIBRATION_TRACE_LONG, indent=2))
