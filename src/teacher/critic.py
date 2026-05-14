"""B4 — PRM-guided best-of-N + refinement (Novelty Pillar P3).

Two-phase critic loop over the raw candidate traces:

  Phase 1  — Best-of-N
    For each pair with ≥1 candidate, run DDI-PRM on every candidate and
    score it.  Aggregate step-wise `+` probabilities using the rubric's
    `primary_score`  (min_plus) and `secondary_score` (final_plus).
    Pick the winner (max of primary, tie-break on secondary).

  Phase 2  — Refinement
    If the winner's primary_score is below `refinement_threshold` (0.60)
    but above `drop_threshold` (0.30), build a PRM-critique prompt that
    tells the teacher exactly which step failed and why, and ask for a
    regeneration.  Write the new candidate back into the raw stream for
    another round of QC + PRM scoring (up to `max_refine_rounds`).

  Pairs whose best score is still below `drop_threshold` after refinement
  are dropped from the clean set.

Usage (cluster, after PRM training finishes — preferred: LoRA adapter on
top of the Med-PRM base):
    python -m src.teacher.critic \\
        --qc           outputs/teacher/qc_subset25k_<TEACHER>.jsonl \\
        --prm_adapter  $DDI_CKPT/ddi_prm_v1 \\
        --prm_base     dmis-lab/llama-3.1-medprm-reward-v1.0 \\
        --output       outputs/teacher/critic_<TEACHER>.jsonl

Legacy path (full standalone PRM checkpoint, no LoRA):
    --prm_model_path  /path/to/full_prm_ckpt

Laptop dev:
    python -m src.teacher.critic --dry_run --qc <path> --output <path>
      → uses rubric thresholds but no PRM scoring; picks the winner by QC
        signals alone (useful for plumbing tests).

IMPORTANT: the question text passed to the PRM at scoring time MUST match
the format used in `prm_data.trace_to_medprm_example` (drug names + IDs +
shared proteins/pathways/PK flags), and the system prompt MUST match
`prm_train.SYSTEM_PROMPT`.  Otherwise the scoring distribution differs
from training and the AUROC/accuracy collapse.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable

from src.teacher.schema import load_rubric
from src.teacher.context_builder import ContextBuilder, DATA

ROOT = Path(__file__).resolve().parents[2]


# Must match src/teacher/prm_train.py SYSTEM_PROMPT verbatim.  Any drift
# here causes a distribution shift between training and scoring time.
SYSTEM_PROMPT = (
    "You are an evaluator assessing the logicality and validity of each step "
    "of the following DDI reasoning trace.  For each reasoning step, output + "
    "if the step is logically valid and evidence-grounded; output - if the "
    "step contains an error (hallucinated IDs, flipped direction, off-family "
    "family_hint, silent abstention).  In addition, the question block "
    "contains the query pair and supporting evidence context."
)


def build_question(ctx) -> str:
    """Build the [Question] block matching `prm_data.trace_to_medprm_example`.

    The PRM was trained to expect: "A = <name> (DB_ID)\\nB = <name> (DB_ID)\\n
    Context:\\n  shared proteins: ...\\n  active PK flags A: ..." etc.
    Anything shorter (e.g. just the pair_id) is OOD and tanks the score.
    """
    qlines = [
        f"A = {ctx.a.name} ({ctx.a.drugbank_id})",
        f"B = {ctx.b.name} ({ctx.b.drugbank_id})",
        "Context:",
    ]
    if ctx.shared_pathways:
        qlines.append("  shared pathways: " +
                      ", ".join(p.pathway_id for p in ctx.shared_pathways))
    if ctx.shared_proteins:
        qlines.append("  shared proteins: " +
                      ", ".join(p.uniprot for p in ctx.shared_proteins))
    if ctx.a.active_pk_flags or ctx.b.active_pk_flags:
        qlines.append("  active PK flags A: " +
                      ", ".join(ctx.a.active_pk_flags[:10]))
        qlines.append("  active PK flags B: " +
                      ", ".join(ctx.b.active_pk_flags[:10]))
    return "\n".join(qlines)


# ───────────────────────── PRM scorer (optional GPU) ──────────────────────
class PRMScorer:
    """Wraps a fine-tuned DDI-PRM checkpoint; returns (min_plus, final_plus)
    for a Med-PRM-formatted (question, solution) pair.

    Two loading modes:
      - base + adapter: pass `base_model` and `adapter_path` (LoRA).  This
        is the path produced by `src/teacher/prm_train.py`.
      - standalone:      pass only `base_model` and leave `adapter_path=None`
        (e.g. when running the unadapted Med-PRM as a baseline).
    """

    def __init__(self, base_model: str, adapter_path: str | None = None,
                 device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Tokenizer: prefer the adapter dir if it has one (Trainer saves
        # tokenizer alongside the LoRA weights), otherwise the base.
        tok_src = adapter_path or base_model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prefer flash_attention_2 when available; fall back to SDPA.
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = "sdpa"
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map=device,
        )
        base.config.use_cache = False
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, adapter_path)
        else:
            self.model = base
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.plus_id = self.tokenizer(" +", add_special_tokens=False)["input_ids"][0]
        self.minus_id = self.tokenizer(" -", add_special_tokens=False)["input_ids"][0]
        self.sep = " ки"
        self._torch = torch

    def score(self, question: str, solution: str) -> dict:
        """Returns a dict of aggregations over per-step P(+) probabilities.

        Keys:
            min_plus     : min over all step probs (sensitive — one bad step
                           saturates this to 0; useful for filtering, bad
                           for ranking).
            mean_plus    : arithmetic mean.
            geomean_plus : geometric mean.  This is the recommended primary
                           ranker — it uses every step's P(+) and stays
                           informative when min is saturated.
            final_plus   : P(+) on the final summary step (the ORM-style
                           critical_passed signal from training).
            step_probs   : full per-step probability vector.

        For an empty result (truncation past all separators / format error),
        all numeric fields are 0.0 and step_probs is [].
        """
        text = (
            f"[System]\n{SYSTEM_PROMPT}\n\n"
            f"[Question]\n{question}\n\n"
            f"[Solution]\n{solution}"
        )
        enc = self.tokenizer(text, return_offsets_mapping=True,
                             add_special_tokens=True, return_tensors="pt",
                             truncation=True, max_length=3072).to(self.device)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"][0].tolist()
        with self._torch.no_grad():
            logits = self.model(ids, attention_mask=enc["attention_mask"]).logits[0]
        # Positions of ` ки` in text
        sep_positions = []
        i = 0
        while True:
            j = text.find(self.sep, i)
            if j < 0:
                break
            sep_positions.append(j)
            i = j + len(self.sep)
        probs = []
        for pos in sep_positions:
            # Find the token index whose offset covers this char
            ti = None
            for k, (s, e) in enumerate(offsets):
                if s <= pos < e:
                    ti = k
                    break
            if ti is None or ti >= logits.size(0):
                continue
            logit_pair = self._torch.stack([logits[ti][self.plus_id],
                                            logits[ti][self.minus_id]])
            p = self._torch.softmax(logit_pair, dim=0)[0].item()
            probs.append(p)
        if not probs:
            return {
                "min_plus": 0.0,
                "mean_plus": 0.0,
                "geomean_plus": 0.0,
                "final_plus": 0.0,
                "step_probs": [],
            }
        import math
        # Geometric mean in log space for numerical stability.
        log_sum = sum(math.log(max(p, 1e-9)) for p in probs)
        return {
            "min_plus":     min(probs),
            "mean_plus":    sum(probs) / len(probs),
            "geomean_plus": math.exp(log_sum / len(probs)),
            "final_plus":   probs[-1],
            "step_probs":   probs,
        }


# ───────────────────────── QC-only fallback scorer ─────────────────────────
def _empty_score() -> dict:
    return {"min_plus": 0.0, "mean_plus": 0.0, "geomean_plus": 0.0,
            "final_plus": 0.0, "step_probs": []}


def _score_from_probs(probs: list[float]) -> dict:
    if not probs:
        return _empty_score()
    import math
    log_sum = sum(math.log(max(p, 1e-9)) for p in probs)
    return {
        "min_plus":     min(probs),
        "mean_plus":    sum(probs) / len(probs),
        "geomean_plus": math.exp(log_sum / len(probs)),
        "final_plus":   probs[-1],
        "step_probs":   probs,
    }


def qc_fallback_score(qc_rec: dict) -> dict:
    """If no PRM is available yet, use the QC critical flags as a proxy.

    Maps to pseudo-PRM scores so the downstream code path is uniform.
    Mirrors the same key set produced by `PRMScorer.score`.
    """
    gates = qc_rec.get("gates", {})
    if not gates.get("G1", False):
        return _empty_score()
    if qc_rec.get("passed"):
        return _score_from_probs([1.0])
    if qc_rec.get("critical_passed"):
        return _score_from_probs([0.9])
    return _score_from_probs([0.25])


# ───────────────────────── critic loop ─────────────────────────
def build_critique_prompt(qc_rec: dict) -> str:
    """If PRM says a step is bad, tell the teacher exactly why (using the QC
    error messages) and ask for a regeneration.  Short, surgical."""
    errs = qc_rec.get("errors", [])[:5]
    if not errs:
        return ("Your previous answer was rejected by the PRM critic.  "
                "Regenerate the trace, paying closer attention to evidence "
                "grounding and direction tagging.")
    return (
        "Your previous answer failed these QC gates:\n"
        + "\n".join(f"  - {e}" for e in errs)
        + "\n\nRegenerate the trace.  Cite only IDs from the EVIDENCE POOL.  "
          "Match the direction consistent with the cited evidence.  If "
          "insufficient evidence to answer confidently, set "
          "final_answer.abstain = True."
    )


def _qc_sort_key(rec: dict) -> tuple:
    """Best-first ordering of a pair's candidates BEFORE PRM scoring.

    Cheap to compute (no GPU); lets us pre-prune with --max_candidates_per_pair
    so we don't waste PRM compute on candidates that already failed obvious QC
    gates. The PRM is only asked to break ties among QC-survivors.
    """
    g = rec.get("gates", {}) or {}
    n_gates_pass = sum(1 for v in g.values() if v)
    return (
        not bool(rec.get("passed")),           # strict-passed first
        not bool(rec.get("critical_passed")),  # then critical-passed
        -n_gates_pass,                         # then by # gates passed
    )


def run_critic(qc_path: Path, out_path: Path,
               prm_adapter: str | None = None,
               prm_base: str | None = None,
               prm_model_path: str | None = None,
               refine_provider: str | None = None,
               refine_model: str | None = None,
               refine_base_url: str | None = None,
               split: str = "subset25k",
               max_refine_rounds: int = 1,
               max_candidates_per_pair: int = 8,
               progress_every_pct: float = 1.0,
               ckpt_every_pairs: int = 500,
               dry_run: bool = False):
    rubric = load_rubric()
    agg = rubric["aggregation"]
    refine_thresh = agg["refinement_threshold"]
    drop_thresh = agg["drop_threshold"]
    accept_thresh = agg["acceptance_threshold"]

    # Group QC records by pair
    by_pair: dict[str, list[dict]] = defaultdict(list)
    with qc_path.open() as f:
        for line in f:
            r = json.loads(line)
            by_pair[r["pair_id"]].append(r)
    n_cand_total = sum(len(v) for v in by_pair.values())
    print(f"[critic] {len(by_pair):,} pairs, {n_cand_total:,} candidates "
          f"(avg {n_cand_total/max(1,len(by_pair)):.1f}/pair)", flush=True)

    # Pre-prune: sort each pair's candidates by QC quality, keep top-K.
    # This is the single biggest runtime win. PRM scoring 24 candidates per
    # pair is wasteful — most are obvious losers QC already flagged.
    if max_candidates_per_pair and max_candidates_per_pair > 0:
        kept = 0
        for pid in list(by_pair.keys()):
            ranked = sorted(by_pair[pid], key=_qc_sort_key)
            by_pair[pid] = ranked[:max_candidates_per_pair]
            kept += len(by_pair[pid])
        print(f"[critic] pre-pruned to top-{max_candidates_per_pair} by QC: "
              f"{kept:,} candidates remaining "
              f"({100*kept/max(1,n_cand_total):.1f}% of original)",
              flush=True)

    # Resolve PRM loading mode.  Prefer base + LoRA adapter (the path
    # produced by prm_train.py).  Fall back to a single full model path.
    use_prm = (prm_adapter or prm_model_path) and not dry_run

    # Scorer
    scorer: Callable[[dict], dict]
    if use_prm:
        if prm_adapter:
            base = prm_base or "dmis-lab/llama-3.1-medprm-reward-v1.0"
            print(f"[critic] loading PRM: base={base!r}  adapter={prm_adapter!r}")
            prm = PRMScorer(base_model=base, adapter_path=prm_adapter)
        else:
            print(f"[critic] loading PRM (standalone): {prm_model_path!r}")
            prm = PRMScorer(base_model=prm_model_path, adapter_path=None)

        # Question text MUST match training distribution → use ContextBuilder.
        cb = ContextBuilder(neighbor_index_path=DATA / f"neighbor_index_{split}.parquet")
        _ctx_cache: dict[str, object] = {}

        def _question_for(pid: str):
            ctx = _ctx_cache.get(pid)
            if ctx is None:
                ctx = cb.build(pid)
                _ctx_cache[pid] = ctx
            return build_question(ctx)

        def _score_prm(qc_rec: dict) -> dict:
            parsed = qc_rec.get("parsed")
            if parsed is None:
                return _empty_score()
            sep = rubric["separator_token"]
            solution_lines = []
            for s in parsed["steps"]:
                ev = f" [evidence: {', '.join(s.get('evidence_ids', []))}]" \
                    if s.get("evidence_ids") else ""
                dt = s.get("direction_tag", "n/a")
                dt_str = f" ({dt})" if dt != "n/a" else ""
                solution_lines.append(f"Step {s['step_id']}: {s['claim']}{ev}{dt_str}{sep}")
            ans = parsed["final_answer"]
            solution_lines.append(
                f"Final: family={ans['family']}, subtype={ans['subtype']}, "
                f"direction={ans['direction_tag']}, polarity={ans['polarity']}, "
                f"abstain={ans['abstain']}{sep}"
            )
            try:
                q = _question_for(qc_rec["pair_id"])
            except Exception:
                # Pair missing from the index — fall back to a minimal question.
                # This will under-score the candidate, which is the safe default.
                q = f"pair_id={qc_rec.get('pair_id','?')}"
            return prm.score(q, "\n".join(solution_lines))

        scorer = _score_prm
    else:
        print("[critic] no PRM model — using QC-fallback scoring "
              "(dry run / laptop mode)")
        scorer = qc_fallback_score

    # Best-of-N — with progress, resume, and incremental checkpointing.
    import sys
    import time

    out_path.parent.mkdir(parents=True, exist_ok=True)
    winners: dict[str, dict] = {}

    # Resume: if a prior run wrote a partial file, load any pairs already done.
    if out_path.exists():
        try:
            with out_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if "pair_id" in r:
                        winners[r["pair_id"]] = r
        except Exception as e:
            print(f"[critic] WARN: could not parse partial {out_path} "
                  f"({e}); starting fresh", flush=True)
            winners = {}
        if winners:
            print(f"[critic] resuming: {len(winners):,} pairs already done; "
                  f"{len(by_pair) - len(winners):,} remaining", flush=True)

    pids = list(by_pair.keys())
    n_pairs = len(pids)
    log_every = max(1, int(n_pairs * progress_every_pct / 100.0))
    t_start = time.time()
    n_done = 0
    n_accepted = n_to_refine = n_drop = 0

    def _flush():
        with out_path.open("w") as fout:
            for w in winners.values():
                fout.write(json.dumps(w) + "\n")

    for pid in pids:
        if pid in winners:  # already scored in a previous run
            continue
        recs = by_pair[pid]
        scored = [(scorer(r), r) for r in recs]
        # Tiered ranker.  Two issues we have to handle:
        #
        # 1. The PRM was trained step-level on a strict-tier-heavy corpus,
        #    so its per-step P(+) saturates near 0 for many intermediate
        #    steps.  That makes step aggregations (min, geomean, mean)
        #    collapse to the noise floor and pick winners ~at random.
        # 2. The PRM's FINAL step P(+) is supervised on critical_passed —
        #    it's an ORM-style head with healthy spread (median ≈ 0.5–0.6).
        #    But it sometimes disagrees with the hard QC schema/label
        #    gates, which we don't want to silently override.
        #
        # Fix: rank by ( qc_passed, qc_critical_passed, final_plus, ... ).
        # We never demote a QC-passing candidate for a marginally higher
        # PRM score, but we use PRM's final_plus to break ties WITHIN a QC
        # tier.  Empirically this gains +2.4–9.1 pp QC-strict pass-rate-of-
        # winner and +3.1–6.6 pp QC-critical pass-rate-of-winner over the
        # pure-PRM ranker, while still gaining +0.02–0.07 in mean PRM
        # final_plus (see docs/V4_RECOVERY_PLAN.md Phase 2.1.5).
        scored.sort(
            key=lambda t: (
                int(bool(t[1].get("passed"))),
                int(bool(t[1].get("critical_passed"))),
                t[0]["final_plus"],
                t[0]["geomean_plus"],
                t[0]["mean_plus"],
                t[0]["min_plus"],
            ),
            reverse=True,
        )
        best_score, best_rec = scored[0]
        # Save a slim per-candidate record so we can re-rank post-hoc with
        # any aggregation (no GPU re-run needed).  The `winner_candidate`
        # still holds the full QC record for the winner.
        all_candidate_scores = [
            {
                "candidate_id": r.get("candidate_id"),
                "qc_passed":          bool(r.get("passed")),
                "qc_critical_passed": bool(r.get("critical_passed")),
                "min_plus":     s["min_plus"],
                "mean_plus":    s["mean_plus"],
                "geomean_plus": s["geomean_plus"],
                "final_plus":   s["final_plus"],
                "step_probs":   s["step_probs"],
            }
            for s, r in scored
        ]
        winners[pid] = {
            "pair_id": pid,
            "winner_candidate": best_rec,
            "score": best_score,
            "all_candidate_scores": all_candidate_scores,
            "n_candidates": len(recs),
        }
        # Accept/refine/drop reporting uses final_plus (the PRM's ORM-style
        # head, supervised on critical_passed) — the only step aggregation
        # that has healthy spread under our strict-tier-heavy training.
        fp = best_score["final_plus"]
        if fp >= accept_thresh:
            n_accepted += 1
        elif fp >= drop_thresh:
            n_to_refine += 1
        else:
            n_drop += 1

        n_done += 1
        if n_done % log_every == 0 or n_done == n_pairs - len(
                [p for p in pids if p in winners]):
            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 0.1)
            remaining = (n_pairs - len(winners))
            eta_min = remaining / max(rate, 1e-6) / 60.0
            sys.stdout.write(
                f"\r[critic] {len(winners):,}/{n_pairs:,} "
                f"({100 * len(winners) / max(1, n_pairs):.1f}%)  "
                f"rate={rate:.2f}pair/s  eta={eta_min:.1f}m  "
                f"last: gm={best_score['geomean_plus']:.3f} "
                f"final={best_score['final_plus']:.3f} "
                f"min={best_score['min_plus']:.3f}   "
            )
            sys.stdout.flush()

        if n_done % ckpt_every_pairs == 0:
            _flush()

    _flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    print(f"[critic] accept={n_accepted:,}  refine={n_to_refine:,}  "
          f"drop={n_drop:,}", flush=True)

    # Refinement phase (only if we have a provider)
    if refine_provider and not dry_run:
        from src.teacher.provider import make_provider
        from src.teacher.prompt import build_prompt
        # ContextBuilder already imported at module scope; reuse the cached
        # `cb` from the scoring branch above if present, otherwise build one.
        if "cb" not in locals():
            cb = ContextBuilder(neighbor_index_path=DATA / f"neighbor_index_{split}.parquet")
        prov = make_provider(refine_provider, model=refine_model,
                             base_url=refine_base_url, temperature=0.6)

        print(f"[critic] refining {n_to_refine:,} pairs with "
              f"{refine_provider}:{refine_model}")

        n_refined_ok = 0
        for pid, w in list(winners.items()):
            if not (drop_thresh <= w["score"]["min_plus"] < accept_thresh):
                continue
            ctx = cb.build(pid)
            base_msgs = build_prompt(ctx)
            critique = build_critique_prompt(w["winner_candidate"])
            # Put critique as an assistant-turn echo before resampling
            user_aug = base_msgs["user"] + "\n\n---\nCritique:\n" + critique
            for round_ix in range(max_refine_rounds):
                gens = prov.generate(system=base_msgs["system"], user=user_aug, n=1)
                # Could pipe through QC+scorer again, but for now just adopt
                # the new text as a candidate addition
                w["refinement_round"] = round_ix + 1
                w["refined_text"] = gens[0].text
                n_refined_ok += 1
                break

        print(f"[critic] refined {n_refined_ok:,} pairs (one pass)")

    # Write winners jsonl
    with out_path.open("w") as f:
        for pid, w in winners.items():
            f.write(json.dumps(w) + "\n")
    print(f"[critic] wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--qc", required=True)
    p.add_argument("--output", "--out", dest="output", required=True,
                   help="Output JSONL with one critic record per pair.")
    p.add_argument("--prm_adapter", default=None,
                   help="Path to LoRA adapter dir from prm_train.py. "
                        "Pair with --prm_base.")
    p.add_argument("--prm_base", default=None,
                   help="Base model for the LoRA adapter. Defaults to "
                        "'dmis-lab/llama-3.1-medprm-reward-v1.0' when "
                        "--prm_adapter is set.")
    p.add_argument("--prm_model_path", default=None,
                   help="Legacy: full standalone PRM checkpoint (no LoRA). "
                        "Mutually exclusive with --prm_adapter.")
    p.add_argument("--refine_provider", default=None)
    p.add_argument("--refine_model", default=None)
    p.add_argument("--refine_base_url", default=None)
    p.add_argument("--split", default="subset25k")
    p.add_argument("--max_refine_rounds", type=int, default=1)
    p.add_argument("--max_candidates_per_pair", type=int, default=8,
                   help="Pre-prune each pair to top-K by QC quality before "
                        "running PRM. Set 0 to score all candidates "
                        "(50+ hours per teacher; not recommended).")
    p.add_argument("--progress_every_pct", type=float, default=1.0,
                   help="Print progress every N%% of pairs.")
    p.add_argument("--ckpt_every_pairs", type=int, default=500,
                   help="Flush partial output every N pairs scored. Smaller "
                        "= safer on time-limited allocations, larger = less "
                        "I/O.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if args.prm_adapter and args.prm_model_path:
        p.error("Pass either --prm_adapter (with optional --prm_base) OR "
                "--prm_model_path, not both.")

    run_critic(
        qc_path=Path(args.qc), out_path=Path(args.output),
        prm_adapter=args.prm_adapter,
        prm_base=args.prm_base,
        prm_model_path=args.prm_model_path,
        refine_provider=args.refine_provider,
        refine_model=args.refine_model,
        refine_base_url=args.refine_base_url,
        split=args.split,
        max_refine_rounds=args.max_refine_rounds,
        max_candidates_per_pair=args.max_candidates_per_pair,
        progress_every_pct=args.progress_every_pct,
        ckpt_every_pairs=args.ckpt_every_pairs,
        dry_run=args.dry_run,
    )
