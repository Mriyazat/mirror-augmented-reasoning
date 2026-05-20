"""Multi-decode + DDI-PRM re-ranking for the student.

For each input record, sample N candidate traces with temperature > 0, score
each trace with the already-trained DDI-PRM, and emit the arg-max-PRM trace
as the final prediction.  This is the consensus-pipeline trick (which gave a
+12.3 pp lift over the best single teacher) applied at inference time.

Why this exists
---------------
The greedy / N=1 decode locks the student onto its single most-confident
hypothesis, and on the held-out distribution that hypothesis is biased
toward the AdverseRisk attractor.  Sampling N traces and re-ranking with
DDI-PRM lets the student commit to a different (correct) hypothesis when
the greedy one would be PRM-low.  The DDI-PRM was trained precisely to
score reasoning quality and is known to improve over best-single-teacher
on the consensus pipeline; the same lift should transfer to inference.

Output schema
-------------
Same as `src/inference/predict.py`, with three extra fields:
    {
      ... (same as predict.py output) ...,
      "rerank": {
        "n_candidates":     int,
        "chosen_idx":       int,                  # which sample won
        "chosen_prm":       float,                # winner geomean_plus
        "candidates": [                           # all N decode traces
          {"raw_output": str, "parse_ok": bool,
           "family": str, "subtype": str, "direction_tag": str,
           "prm_geomean": float, "prm_min": float, "prm_final": float}
        ],
      }
    }

Usage
-----
    python -m src.inference.predict_with_rerank \
        --adapter   $DDI_CKPT/student/ddi_v4_best_phase4_prm_dpo_macro0797 \
        --prm_base  dmis-lab/llama-3.1-medprm-reward-v1.0 \
        --prm_adapter $DDI_CKPT/ddi_prm_v1 \
        --input     outputs/eval_prompts/random_full_test_5000_stratified.with_neighbors.prompts.jsonl \
        --output    outputs/eval_prompts/pred_${RUN}_rerank_n4_ab.jsonl \
        --n_samples 4 \
        --temperature 0.7 \
        --batch     4 \
        --max_new_tokens 768

    # Mirror pass:
    python -m src.inference.predict_with_rerank \
        ... --mirror \
        --output outputs/eval_prompts/pred_${RUN}_rerank_n4_ba.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.inference.predict import (
    _build_prompt_messages,
    _extract_json,
    _normalize_trace,
    _get_context_ids,
    _CTX_STATS,
    _MIRROR_STATS,
)
from src.teacher.critic import PRMScorer
from src.teacher.schema import load_rubric


# ----------------------------------------------------------------------
# Med-PRM input shaping (mirrors src/teacher/score_traces_prm.py but
# operates on a parsed dict that came straight from the student decode).
# ----------------------------------------------------------------------
def _build_medprm_solution(parsed: dict, sep: str) -> str | None:
    if not isinstance(parsed, dict) or "final_answer" not in parsed:
        return None
    sol_lines: list[str] = []
    for s in parsed.get("steps", []) or []:
        sid = s.get("step_id")
        role = s.get("role", "unknown")
        claim = (s.get("claim") or "").strip()
        ev = (s.get("evidence_ids") or [])
        dt = s.get("direction_tag", "n/a")
        ev_str = f" [evidence: {', '.join(ev)}]" if ev else ""
        dt_str = f" ({dt})" if dt and dt != "n/a" else ""
        sol_lines.append(f"Step {sid}: {claim}{ev_str}{dt_str}{sep}")
        _ = role  # kept for parity with score_traces_prm
    ans = parsed["final_answer"]
    sol_lines.append(
        f"Final: family={ans.get('family','')}, subtype={ans.get('subtype','')}, "
        f"direction={ans.get('direction_tag','')}, "
        f"polarity={ans.get('polarity','')}, "
        f"abstain={ans.get('abstain', False)}{sep}"
    )
    return "\n".join(sol_lines)


# ----------------------------------------------------------------------
# Generation helpers
# ----------------------------------------------------------------------
def _load_model(base_model: str, adapter_dir: str | None,
                dtype_str: str, device_map: str):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_str]
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=dtype, trust_remote_code=True, device_map=device_map,
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, trust_remote_code=True, device_map=device_map,
        )
    model = PeftModel.from_pretrained(base, adapter_dir) if adapter_dir else base
    model.eval()
    return model, tokenizer


def _sample_n(model, tokenizer, prompt: str, n: int, max_new_tokens: int,
              temperature: float, top_p: float, batch: int) -> list[str]:
    """Generate `n` independent samples for one prompt.  Re-tokenises the
    prompt `ceil(n/batch)` times and concatenates."""
    enc = tokenizer([prompt], return_tensors="pt", padding=True,
                    truncation=True, max_length=4096)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[1]
    outs: list[str] = []
    remaining = n
    while remaining > 0:
        k = min(batch, remaining)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        gen = out[:, input_len:]
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        remaining -= k
    return outs


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_samples", type=int, default=4,
                    help="Number of decodes per record (the N in N-best).")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--batch", type=int, default=4,
                    help="num_return_sequences per generate() call. "
                         "Total decodes per record = n_samples; the model "
                         "runs ceil(n_samples / batch) generate() calls.")
    ap.add_argument("--torch_dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", default="cuda:0")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--input_order", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--with_context", dest="with_context", action="store_true")
    ap.add_argument("--no_context", dest="with_context", action="store_false")
    ap.set_defaults(with_context=True)

    # PRM scorer
    ap.add_argument("--prm_base",
                    default="dmis-lab/llama-3.1-medprm-reward-v1.0",
                    help="Med-PRM base for the DDI-PRM.")
    ap.add_argument("--prm_adapter", default=None,
                    help="LoRA adapter for the DDI-PRM.  Omit to use the "
                         "Med-PRM seed unmodified.")
    ap.add_argument("--prm_device", default="cuda:0",
                    help="GPU for the PRM.  Use cuda:1 if you have a second "
                         "GPU; otherwise share cuda:0 with the student.")
    ap.add_argument("--rerank_metric", default="geomean_plus",
                    choices=["geomean_plus", "min_plus", "mean_plus", "final_plus"])

    args = ap.parse_args()

    input_order = args.input_order or ("ba" if args.mirror else "ab")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- load student ----
    print(f"[rerank] loading student from {args.adapter}")
    s_model, s_tok = _load_model(
        args.base_model, args.adapter, args.torch_dtype, args.device_map,
    )

    # ---- load PRM ----
    print(f"[rerank] loading DDI-PRM from base={args.prm_base} "
          f"adapter={args.prm_adapter} on {args.prm_device}")
    scorer = PRMScorer(args.prm_base, args.prm_adapter, args.prm_device)
    rubric = load_rubric()
    sep = rubric["separator_token"]

    # ---- iterate input ----
    records = []
    with open(args.input) as fin:
        for line in fin:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit > 0:
        records = records[:args.limit]

    # ---- resume bookkeeping ----
    open_mode = "w"
    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        try:
            with out_path.open() as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pid = r.get("pair_id")
                    if pid:
                        done_ids.add(pid)
            open_mode = "a"
            records = [r for r in records if r.get("pair_id") not in done_ids]
            print(f"[rerank] --resume: kept {len(done_ids)} prior records, "
                  f"{len(records)} still to predict.")
        except Exception as e:
            print(f"[rerank] --resume: failed to read {out_path} ({e}); overwriting.")

    print(f"[rerank] {len(records)} input records, "
          f"n_samples={args.n_samples}, temp={args.temperature}, "
          f"mirror={args.mirror}")

    n_parse_ok = 0
    n_parse_fail_all = 0
    n_picked_top = 0
    t0 = time.time()
    with out_path.open(open_mode) as fout:
        for ridx, rec in enumerate(records):
            msgs = _build_prompt_messages(rec, args.mirror)
            prompt = s_tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            user_msg_text = next(
                (m.get("content", "") for m in msgs if m.get("role") == "user"),
                "",
            )

            # ---- generate N candidates ----
            samples = _sample_n(
                s_model, s_tok, prompt, args.n_samples,
                args.max_new_tokens, args.temperature, args.top_p,
                args.batch,
            )

            # ---- score each candidate with DDI-PRM ----
            cand_records = []
            best_idx = -1
            best_key: tuple | None = None
            best_score = 0.0
            best_parsed: dict | None = None
            best_raw = ""
            best_err: str | None = None
            zero_scored = {
                "min_plus": 0.0, "mean_plus": 0.0,
                "geomean_plus": 0.0, "final_plus": 0.0,
                "step_probs": [],
            }
            for cidx, raw in enumerate(samples):
                parsed, err = _extract_json(raw)
                parse_ok = parsed is not None
                fa = (parsed or {}).get("final_answer", {}) or {}
                scored = zero_scored
                if parse_ok:
                    sol = _build_medprm_solution(parsed, sep)
                    if sol is not None:
                        scored = scorer.score(user_msg_text, sol)
                metric_val = float(scored.get(args.rerank_metric, 0.0))

                cand_records.append({
                    "raw_output":    raw,
                    "parse_ok":      parse_ok,
                    "family":        fa.get("family"),
                    "subtype":       fa.get("subtype"),
                    "direction_tag": fa.get("direction_tag"),
                    "prm_geomean":   float(scored.get("geomean_plus", 0.0)),
                    "prm_mean":      float(scored.get("mean_plus", 0.0)),
                    "prm_min":       float(scored.get("min_plus", 0.0)),
                    "prm_final":     float(scored.get("final_plus", 0.0)),
                })

                # Prefer parseable candidates; break ties by selected PRM
                # metric; finally earliest-index wins for determinism.
                key = (1 if parse_ok else 0, metric_val, -cidx)
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = cidx
                    best_score = metric_val
                    best_parsed = parsed
                    best_raw = raw
                    best_err = err

            if best_idx == 0:
                n_picked_top += 1

            if best_parsed is None:
                best_parsed = {}
                n_parse_fail_all += 1
            else:
                n_parse_ok += 1

            shaped = _normalize_trace(best_parsed, rec)
            pair_id = rec.get("pair_id", "")
            ctx_ids = rec.get("context_ids") or []
            if args.with_context and not ctx_ids:
                ctx_ids = _get_context_ids(pair_id)

            out_rec = {
                "pair_id":          pair_id,
                "input_order":      input_order,
                "context_ids":      ctx_ids,
                "trace":            shaped["trace"],
                "final_prediction": shaped["final_prediction"],
                "raw_output":       best_raw,
                "parse_ok":         best_err is None,
                "parse_error":      best_err,
                "rerank": {
                    "n_candidates": args.n_samples,
                    "metric":       args.rerank_metric,
                    "chosen_idx":   best_idx,
                    "chosen_prm":   float(best_score),
                    "candidates":   cand_records,
                },
            }
            fout.write(json.dumps(out_rec) + "\n")

            done = ridx + 1
            if done % 20 == 0 or done == len(records):
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                print(f"[rerank] {done}/{len(records)} "
                      f"parse_ok={n_parse_ok} all_fail={n_parse_fail_all} "
                      f"picked_idx0={n_picked_top} "
                      f"({rate:.2f} rec/s)")

    print(f"[rerank] done. parse_ok={n_parse_ok} "
          f"all_fail={n_parse_fail_all} picked_idx0={n_picked_top} "
          f"out={out_path}")
    if args.with_context:
        print(f"[rerank] context_ids stats: {_CTX_STATS}")
    if args.mirror:
        print(f"[rerank] mirror stats: {_MIRROR_STATS}")


if __name__ == "__main__":
    main()
