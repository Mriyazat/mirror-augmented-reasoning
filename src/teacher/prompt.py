"""B1b — Teacher prompt builder.

Compiles a ContextBundle into the exact system + user messages the teacher
LLM consumes.  Output prompt is deterministic and diffable across versions
(stored with every generated candidate for reproducibility).

Design notes:
  - The prompt is aggressive about evidence grounding ("cite IDs from the
    EVIDENCE POOL section only; hallucinated IDs will be rejected").
  - The prompt shows `EXAMPLE_JSON_SKELETON` verbatim so the teacher always
    produces the schema B2 expects.
  - We do NOT tell the teacher the gold label — the teacher must reason
    from the context alone.  (This is teacher-as-reasoner, not
    teacher-knows-answer.)
  - Direction is explicitly scaffolded: the prompt pre-declares which drug
    is A and which is B, and asks the teacher to emit direction_tag for
    every non-trivial step.
  - The prompt VERSION string lives in `PROMPT_VERSION` below and its SHA
    is exposed by `prompt_sha()` — every generated record stores these so
    we can always re-derive the exact prompt used.
"""
from __future__ import annotations

import hashlib

from src.teacher.context_builder import ContextBundle
from src.teacher.schema import EXAMPLE_JSON_SKELETON, load_rubric

# ---------------------------------------------------------------------------
# Prompt versioning (bump on any user-visible change to the prompt wording)
# ---------------------------------------------------------------------------
PROMPT_VERSION = "v4.3.1"

FAMILY_DEFINITIONS = """\
family definitions (you must pick exactly one for `final_answer.family`):
  • PK_Metabolism     — CYP/enzyme-mediated metabolism change
  • PK_Excretion      — renal/biliary clearance change (OAT/OCT/OATP/BCRP)
  • PK_Absorption     — GI absorption / P-gp-mediated absorption change
  • PK_Distribution   — protein-binding / serum concentration change
  • PD_Activity       — pharmacodynamic synergy / antagonism (no PK shift)
  • Efficacy          — therapeutic efficacy altered (non-adverse)
  • AdverseRisk       — increased risk of adverse event; polarity can be
                        `risk` (risk up) or `risk_down` (protective)
"""

# Per-family *subtype whitelist* — top-20 subtypes covering ≥93% of the mass
# in labels_hierarchical.parquet.  Picking a subtype OUTSIDE this list is
# almost certainly wrong.  (Computed once from Phase A; refresh on re-run.)
SUBTYPE_VOCAB = {
    "AdverseRisk": [
        "adverse_effects", "cns_depression", "qtc_prolongation",
        "hypertension", "methemoglobinemia", "bleeding", "hypoglycemia",
        "hyperkalemia", "bleeding_and_hemorrhage", "nephrotoxicity",
        "tachycardia", "hyperglycemia", "gastrointestinal_irritation",
        "serotonin_syndrome", "gastrointestinal_bleeding",
        "renal_failure_hyperkalemia_and_hypertension",
        "hypotension", "dehydration", "hypokalemia", "infection",
    ],
    "Efficacy":       ["therapeutic_efficacy", "diagnostic_effectiveness"],
    "PD_Activity": [
        "antihypertensive", "hypotensive",
        "central_nervous_system_depressant_cns_depressant",
        "thrombogenic", "arrhythmogenic", "hypoglycemic", "anticoagulant",
        "hyperkalemic", "bradycardic", "sedative", "neuroexcitatory",
        "neurotoxic", "neuromuscular_blocking", "orthostatic_hypotensive",
        "hypertensive", "immunosuppressive", "hypertensive_and_vasoconstricting",
        "serotonergic", "stimulatory", "qtc_prolonging",
    ],
    "PK_Absorption":   ["absorption_change", "bioavailability"],
    "PK_Distribution": ["serum_concentration", "protein_binding",
                        "active_metabolite_serum_conc"],
    "PK_Excretion":    ["excretion_rate"],
    "PK_Metabolism":   ["metabolism"],
}


def _format_subtype_vocab() -> str:
    lines = ["subtype whitelist (pick exactly one from the row matching `final_answer.family`):"]
    for fam in sorted(SUBTYPE_VOCAB):
        vocab = ", ".join(SUBTYPE_VOCAB[fam])
        lines.append(f"  • {fam}: {vocab}")
    return "\n".join(lines)


def format_pk_flags(flags: list[str], drug_label: str) -> str:
    if not flags:
        return f"  {drug_label}: (no CYP/P-gp/OATP/BCRP activity on file)"
    by_kind = {"_inh": [], "_ind": [], "_sub": []}
    for f in flags:
        for k in by_kind:
            if f.endswith(k):
                name = f[:-len(k)].upper().replace("_", "")
                by_kind[k].append(name)
                break
    parts = []
    if by_kind["_inh"]:
        parts.append("inhibits " + ", ".join(by_kind["_inh"]))
    if by_kind["_ind"]:
        parts.append("induces " + ", ".join(by_kind["_ind"]))
    if by_kind["_sub"]:
        parts.append("substrate of " + ", ".join(by_kind["_sub"]))
    return f"  {drug_label}: {' ; '.join(parts)}"


def build_system_prompt() -> str:
    rubric = load_rubric()
    tg = rubric["teacher_generation"]
    summary_max = rubric.get("summary_constraints", {}).get("max_words", 80)
    return f"""You are a DDI mechanism expert producing step-wise reasoning traces for a drug-drug-interaction pair.

For each query pair of drugs, you must:
  1. Reason step-by-step using ONLY the EVIDENCE POOL provided below.
  2. Cite evidence as IDs (DrugBank IDs, UniProt IDs, pathway IDs SMPDB:*/KEGG:*, ATC codes, PK-flag tokens like `cyp3a4_inh`, or neighbor `pair_id`s). Every ID you cite must appear verbatim in the EVIDENCE POOL.
  3. Tag direction on each step: `a_to_b` (A acts on B), `b_to_a` (B acts on A), `bidirectional`, or `n/a` (pure mechanism description with no directional claim).
  4. Finish with a conclusion step and a `final_answer` JSON object that INCLUDES a ≤{summary_max}-word natural-language `summary`.

DIRECTION SEMANTICS (read carefully — direction errors are the #1 failure mode):
  • "A" is the first drug listed in QUERY PAIR; "B" is the second drug.
  • `a_to_b` means A's presence changes B's behavior (e.g. "A inhibits the enzyme that metabolizes B" → a_to_b).
  • `b_to_a` is the reverse. Use the subject-verb direction in each claim to decide.
  • `bidirectional` is only for genuinely symmetric mechanisms (both drugs compete for the same transporter/enzyme with comparable potency).

{FAMILY_DEFINITIONS}

{_format_subtype_vocab()}

Output format — a single JSON object with two keys (`steps` and `final_answer`), NO markdown, NO prose outside JSON:

```
{EXAMPLE_JSON_SKELETON}
```

STEP ROLE VOCABULARY (v4.3.1 — fixed set; each step's `role` MUST be chosen from this list):
  Evidence-pool inspection:
    • `pathway`             — cite pathway annotations (shared or per-drug).
    • `protein`             — cite protein targets (shared or per-drug).
    • `pk_flag`             — cite CYP/P-gp/OATP/BCRP activity flags.
    • `structural`          — cite SMILES / substructural similarity.
    • `atc`                 — cite ATC-class overlap.
    • `mechanism_of_action` — narrate a drug's MoA from its DrugBank record.
  Retrieval / analogy:
    • `neighbor_pair`       — cite a specific labeled neighbor pair as analogy.
    • `pair_similarity`     — cite jaccard / tanimoto / atc-prefix scores.
  Meta-evidence (for sparse pairs):
    • `evidence_gap`        — state what is MISSING (no shared pathway, no MoA, etc.).
    • `abstention`          — explicit abstention reasoning (used when the final_answer will set abstain=true).
  Inference:
    • `direction`           — resolve a_to_b vs b_to_a vs bidirectional.
    • `conclusion`          — MUST be the LAST step; summarizes and commits to final_answer.

  Do NOT invent new role names (e.g. `mechanism_description`, `shared_protein`,
  `analogy`, `neighbor`, `evidence_assessment`, `abstention_reasoning`).  Pick
  the closest match from the 12 above. A trace that emits any other role value
  WILL be rejected by the quality-control gate.

Constraints:
  • {tg['min_steps']}–{tg['max_steps']} reasoning steps.
  • {tg['min_tokens_per_step']}–{tg['max_tokens_per_step']} tokens per step (keep each step tight).
  • The last step's `role` MUST be "conclusion".
  • The conclusion's `direction_tag` and `polarity` MUST match `final_answer.direction_tag` / `final_answer.polarity`.
  • `final_answer.family` MUST be from the 7-family list.
  • `final_answer.subtype` MUST be from the whitelist for that family.

SUMMARY FIELD (critical — student models will be trained on this compact path):
  • `final_answer.summary` MUST be a ≤{summary_max}-word natural-language recap (1-2 short sentences).
  • MUST name both drugs (by name) and state the mechanism + direction in plain language.
  • MUST be consistent with the `final_answer.{{family, subtype, direction_tag, polarity}}` fields.
  • Do NOT cite IDs in the summary — it is prose, not evidence. (IDs live in the steps.)
  • When `abstain = true`, the summary should state "insufficient evidence to conclude" with one reason.

DECISIVE LANGUAGE (critical — hedging amplifies from teacher to student):
  When you DO commit to an answer (abstain = false), the summary MUST use decisive language.
  AVOID hedging markers: "may", "might", "possibly", "potentially", "likely", "probably",
  "perhaps", "maybe", "appears to", "seems to", "tends to", "in some cases".
  Instead, use direct causative verbs: "inhibits", "induces", "reduces", "increases",
  "decreases", "raises", "lowers", "competes with", "displaces".
    • GOOD: "Amiodarone inhibits CYP3A4, so simvastatin metabolism decreases and its plasma level rises."
    • BAD:  "Amiodarone may potentially inhibit CYP3A4, which could possibly lead to increased simvastatin levels in some patients."
  Hedging is permitted ONLY when abstain = true (genuine uncertainty about sparse evidence).

ABSTENTION POLICY:
  If the EVIDENCE POOL is too sparse to support a confident answer, set
  `final_answer.abstain = true` and `final_answer.confidence < 0.35`. Specifically:
    • If fewer than 2 of the following are non-empty — (MoA-A, MoA-B, PK-flags-A, PK-flags-B, shared pathways, shared proteins, per-drug pathways, per-drug proteins, neighbors) — you SHOULD abstain.
    • Do NOT fake evidence. Do NOT guess from drug names alone. Abstention is a valid, expected output for data-sparse pairs.

Failure modes we explicitly penalize (every one is auto-detected by our QC):
  • Hallucinating an ID not in the EVIDENCE POOL.                       (G2)
  • Using silencing-language ("I cannot", "insufficient info") mid-chain
    without setting `abstain = true`.                                   (G6)
  • Flipping direction (a_to_b vs b_to_a) without evidence support.     (G3)
  • Emitting a family not in the 7-family list above.                   (G4)
  • Emitting a subtype not in the whitelist for the chosen family.      (G10)
  • Summary that exceeds {summary_max} words or fails to name either drug. (G8)
  • Excessive hedging ("may", "might", "possibly", "potentially", …) in the
    summary when abstain = false.                                       (G9)
"""


def format_neighbors(ctx: ContextBundle, limit: int = 5) -> str:
    if not ctx.neighbors:
        return "  (no mechanistic-neighbor pairs surfaced by retrieval)"
    lines = []
    for n in ctx.neighbors[:limit]:
        pol = f" pol={n.polarity}" if n.polarity else ""
        lines.append(
            f"  • {n.pair_id} : {n.a_name} ↔ {n.b_name} → "
            f"{n.family}/{n.subtype} [{n.direction_tag}{pol}]  sim={n.similarity:.2f}"
        )
    return "\n".join(lines)


def format_shared_proteins(ctx: ContextBundle) -> str:
    if not ctx.shared_proteins:
        return "  (no overlapping drug-protein targets)"
    lines = []
    for p in ctx.shared_proteins:
        a_act = ",".join(p.a_actions) if p.a_actions else "—"
        b_act = ",".join(p.b_actions) if p.b_actions else "—"
        lines.append(
            f"  • {p.uniprot} ({p.protein_name}) — A:{p.a_role}/{a_act}  B:{p.b_role}/{b_act}"
        )
    return "\n".join(lines)


def format_shared_pathways(ctx: ContextBundle) -> str:
    if not ctx.shared_pathways:
        return "  (no overlapping pathways)"
    return "\n".join(
        f"  • {p.pathway_id} ({p.source}) {p.pathway_name}  [{p.category}]"
        for p in ctx.shared_pathways
    )


def format_drug_pathways(pw_list, label: str) -> str:
    if not pw_list:
        return f"  {label}: (no pathway annotations on file)"
    return "\n".join(
        f"  {label} • {p.pathway_id} ({p.source}) {p.pathway_name}"
        for p in pw_list
    )


def format_drug_proteins(pr_list, label: str) -> str:
    if not pr_list:
        return f"  {label}: (no protein targets on file)"
    lines = []
    for p in pr_list:
        act = ",".join(p.actions) if p.actions else "—"
        lines.append(f"  {label} • {p.uniprot} ({p.protein_name}) — {p.role}/{act}")
    return "\n".join(lines)


def build_user_prompt(ctx: ContextBundle) -> str:
    a, b = ctx.a, ctx.b
    a_hl = f"{a.half_life_hours:.1f}h" if a.half_life_hours else "unknown"
    b_hl = f"{b.half_life_hours:.1f}h" if b.half_life_hours else "unknown"
    a_mw = f"{a.mw:.0f}" if a.mw else "unknown"
    b_mw = f"{b.mw:.0f}" if b.mw else "unknown"

    moa_a = a.moa or "(no mechanism_of_action on file)"
    moa_b = b.moa or "(no mechanism_of_action on file)"

    dens = ctx.evidence_density()
    non_empty_pools = sum(1 for k, v in dens.items() if v)
    abstain_hint = (
        "\n⚠ evidence is sparse ({} / {} pools non-empty) — strongly consider abstaining.".format(
            non_empty_pools, len(dens))
        if non_empty_pools < 3 else ""
    )

    return f"""QUERY PAIR
  A = {a.name}  ({a.drugbank_id})  ATC={','.join(a.atc_codes) or '—'}  MW={a_mw}  t½={a_hl}
  B = {b.name}  ({b.drugbank_id})  ATC={','.join(b.atc_codes) or '—'}  MW={b_mw}  t½={b_hl}

EVIDENCE POOL — cite ONLY IDs that appear below.{abstain_hint}

[Drug A mechanism of action]
  {moa_a}

[Drug B mechanism of action]
  {moa_b}

[Active PK flags]
{format_pk_flags(a.active_pk_flags, 'A=' + a.name)}
{format_pk_flags(b.active_pk_flags, 'B=' + b.name)}

[Shared pathways — both drugs affect the same pathway]
{format_shared_pathways(ctx)}

[Per-drug pathways — use when shared is empty]
{format_drug_pathways(ctx.a_pathways, 'A=' + a.name)}
{format_drug_pathways(ctx.b_pathways, 'B=' + b.name)}

[Shared proteins — both drugs bind the same target/enzyme]
{format_shared_proteins(ctx)}

[Per-drug proteins — use when shared is empty]
{format_drug_proteins(ctx.a_proteins, 'A=' + a.name)}
{format_drug_proteins(ctx.b_proteins, 'B=' + b.name)}

[Pair similarity signatures]
  pathway_jaccard = {ctx.pathway_jaccard:.3f}
  protein_jaccard = {ctx.protein_jaccard:.3f}
  smiles_tanimoto = {ctx.smiles_tanimoto:.3f}
  atc_prefix_depth = {ctx.atc_prefix_depth}  (out of 7)

[Top mechanistic-neighbor pairs and their labels — use these as analogical evidence]
{format_neighbors(ctx)}

TASK
  Output the JSON object described in the system prompt.  Reason from evidence only.
"""


def prompt_sha(system: str, user: str) -> str:
    """Stable 16-char hex of (system, user).  Stored on every record."""
    h = hashlib.sha256()
    h.update(b"SYSTEM\n")
    h.update(system.encode("utf-8"))
    h.update(b"\nUSER\n")
    h.update(user.encode("utf-8"))
    return h.hexdigest()[:16]


def build_prompt(ctx: ContextBundle, extra_user_guidance: str | None = None,
                 prompt_version_suffix: str | None = None) -> dict:
    """Return a `messages` dict suitable for chat-completion APIs, with
    reproducibility metadata (version + sha) attached."""
    system = build_system_prompt()
    user = build_user_prompt(ctx)
    version = PROMPT_VERSION
    if extra_user_guidance:
        user = (
            user.rstrip()
            + "\n\n[Additional family-disambiguation guidance]\n"
            + extra_user_guidance.strip()
            + "\n"
        )
        if prompt_version_suffix:
            version = f"{PROMPT_VERSION}+{prompt_version_suffix}"
    return {
        "system": system,
        "user": user,
        "prompt_version": version,
        "prompt_sha": prompt_sha(system, user),
    }
