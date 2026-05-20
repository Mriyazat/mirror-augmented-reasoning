"""Zero-shot DDI prediction with LOCAL HuggingFace models (e.g. BioMistral-7B,
Meditron-7B, Med42-v2-8B, or any other instruction-tuned chat model).

Same prompt + same parser + same JSONL schema as `predict_with_frontier_llm.py`
so the rest of the eval pipeline works unchanged.

Usage:
    python -m src.inference.predict_with_hf_local \\
        --prompts outputs/eval_prompts/pair_cold_test_5000_stratified.with_neighbors.prompts.jsonl \\
        --manifest outputs/eval_prompts/pair_cold_test_5000_stratified.manifest.jsonl \\
        --output  outputs/eval_prompts/pred_biomistral_pair_cold_500.jsonl \\
        --model BioMistral/BioMistral-7B \\
        --n_pairs 500 --stratified --resume

Supported models (any HF chat model with apply_chat_template works):
    BioMistral/BioMistral-7B
    epfl-llm/meditron-7b               (no chat template; use --base_prompt)
    m42-health/Llama3-Med42-8B
    aaditya/Llama3-OpenBioLLM-8B
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
            "PK_Distribution", "PK_Excretion", "PK_Metabolism"]


def _truth() -> dict[str, str]:
    rows = pq.read_table(
        ROOT / "data_processed/labels_hierarchical.parquet",
        columns=["pair_id", "family"],
    ).to_pylist()
    return {r["pair_id"]: r["family"] for r in rows}


def _stratified_sample(pair_ids: list[str], truth: dict[str, str], n: int, seed: int) -> list[str]:
    by_fam: dict[str, list[str]] = defaultdict(list)
    for pid in pair_ids:
        if pid in truth:
            by_fam[truth[pid]].append(pid)
    rng = random.Random(seed)
    for v in by_fam.values():
        rng.shuffle(v)
    per_fam = max(1, n // len(by_fam))
    out: list[str] = []
    for fam, ids in by_fam.items():
        out.extend(ids[:per_fam])
    rng.shuffle(out)
    return out[:n]


# Same parser as predict_with_frontier_llm.py
def parse_trace(raw: str | None) -> dict | None:
    if not raw:
        return None
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    try:
        return json.loads(txt)
    except Exception:
        pass
    txt2 = re.sub(r",\s*([}\]])", r"\1", txt)
    try:
        return json.loads(txt2)
    except Exception:
        pass
    m = re.search(r'"final_answer"\s*:\s*({.*})', txt, re.DOTALL)
    if m:
        candidate = m.group(1)
        depth = 0
        end = None
        for i, ch in enumerate(candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is not None:
            try:
                fa = json.loads(candidate[: end + 1])
                return {"final_answer": fa}
            except Exception:
                pass
    out: dict = {}
    m = re.search(r'"family"\s*:\s*"([A-Za-z_]+)"', txt)
    if m and m.group(1) in FAMILIES:
        out["family"] = m.group(1)
    if not out:
        return None
    sub = re.search(r'"subtype"\s*:\s*"([^"\n]+)"', txt)
    if sub:
        out["subtype"] = sub.group(1)
    dt = re.search(r'"direction_tag"\s*:\s*"([^"\n]+)"', txt)
    if dt:
        out["direction_tag"] = dt.group(1)
    pol = re.search(r'"polarity"\s*:\s*"([^"\n]+)"', txt)
    if pol:
        out["polarity"] = pol.group(1)
    conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', txt)
    if conf:
        try:
            out["confidence"] = float(conf.group(1))
        except Exception:
            pass
    abst = re.search(r'"abstain"\s*:\s*(true|false)', txt)
    if abst:
        out["abstain"] = abst.group(1) == "true"
    return {"final_answer": out}


def trace_to_final_prediction(trace: dict | None) -> dict:
    if not trace or "final_answer" not in trace:
        return {"family": None, "subtype": None, "abstain": True, "confidence": 0.0}
    fa = trace["final_answer"]
    fam = fa.get("family")
    if fam not in FAMILIES:
        return {"family": None, "subtype": None, "abstain": True, "confidence": 0.0}
    return {
        "family": fam,
        "subtype": fa.get("subtype"),
        "direction_tag": fa.get("direction_tag"),
        "polarity": fa.get("polarity"),
        "abstain": bool(fa.get("abstain", False)),
        "confidence": float(fa.get("confidence", 0.5) or 0.5),
        "summary": fa.get("summary"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True, help="HF model id (e.g. BioMistral/BioMistral-7B)")
    ap.add_argument("--torch_dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--n_pairs", type=int, default=500)
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--input_order", default="ab")
    ap.add_argument("--fallback_prompt", action="store_true",
                    help="If model has no chat template, concatenate system + user as a single prompt.")
    ap.add_argument("--use_safetensors", default="auto",
                    choices=["auto", "true", "false"],
                    help="auto = try safetensors first then .bin; explicit overrides.")
    ap.add_argument("--local_files_only", action="store_true",
                    help="Force loading from local HF cache; never query the hub.")
    ap.add_argument("--chat_template", default="auto",
                    choices=["auto", "llama3", "mistral", "none"],
                    help="auto=detect from tokenizer+config; "
                         "llama3=force Llama-3-Instruct template (use for OpenBioLLM, Med42, etc. when missing); "
                         "mistral=force Mistral instruct template; "
                         "none=use raw fallback prompt.")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = set()
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                manifest.add(json.loads(line)["pair_id"])
    truth = _truth()

    prompts: dict[str, list[dict]] = {}
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("pair_id")
            if pid in manifest:
                prompts[pid] = r["messages"]

    candidate_ids = sorted(set(prompts) & set(truth))
    if args.stratified:
        ids = _stratified_sample(candidate_ids, truth, args.n_pairs, args.seed)
    else:
        rng = random.Random(args.seed)
        ids = candidate_ids[:]
        rng.shuffle(ids)
        ids = ids[: args.n_pairs]

    done: set[str] = set()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["pair_id"])
                    except Exception:
                        pass

    todo = [pid for pid in ids if pid not in done]
    print(f"[hf] pairs to score: {len(todo)} (of {len(ids)} sampled, {len(done)} already done)")

    print(f"[hf] loading {args.model}")
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.torch_dtype]

    # Resolve model id -> local snapshot path. This bypasses a transformers
    # bug where AutoModelForCausalLM.from_pretrained(HF_ID, local_files_only=True)
    # returns None for checkpoint_files when the repo only ships pytorch_model.bin.
    model_path = args.model
    if Path(args.model).exists():
        model_path = args.model
        print(f"[hf] using local path {model_path}")
    else:
        try:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(
                args.model, local_files_only=args.local_files_only,
            )
            print(f"[hf] resolved {args.model} -> {model_path}")
        except Exception as e:
            print(f"[hf] snapshot_download failed: {e}; falling back to HF id", file=sys.stderr)
            model_path = args.model

    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=args.local_files_only,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- chat template injection for tokenizers that are missing one ----
    LLAMA3_TEMPLATE = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + "
        "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
        "{{ content }}{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )
    MISTRAL_TEMPLATE = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}{% endfor %}"
    )
    if args.chat_template == "llama3":
        tok.chat_template = LLAMA3_TEMPLATE
        print("[hf] forced chat_template=llama3")
    elif args.chat_template == "mistral":
        tok.chat_template = MISTRAL_TEMPLATE
        print("[hf] forced chat_template=mistral")
    elif args.chat_template == "auto" and not getattr(tok, "chat_template", None):
        # Try to detect architecture from config.json
        try:
            with open(Path(model_path) / "config.json") as f:
                cfg = json.load(f)
            mtype = (cfg.get("model_type") or "").lower()
            archs = " ".join(cfg.get("architectures") or []).lower()
            if "llama" in mtype or "llama" in archs:
                tok.chat_template = LLAMA3_TEMPLATE
                print(f"[hf] auto-detected llama architecture; injected llama3 chat_template")
            elif "mistral" in mtype or "mistral" in archs:
                tok.chat_template = MISTRAL_TEMPLATE
                print(f"[hf] auto-detected mistral architecture; injected mistral chat_template")
        except Exception as e:
            print(f"[hf] chat_template auto-detect failed: {e}", file=sys.stderr)

    def _load(use_safetensors_val):
        kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
        )
        if use_safetensors_val is not None:
            kwargs["use_safetensors"] = use_safetensors_val
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    if args.use_safetensors == "true":
        model = _load(True)
    elif args.use_safetensors == "false":
        model = _load(False)
    else:
        try:
            model = _load(True)
        except Exception as e_safe:
            print(f"[hf] safetensors load failed ({type(e_safe).__name__}: {e_safe}); "
                  f"retrying with use_safetensors=False ...", file=sys.stderr)
            model = _load(False)
    model.eval()

    has_chat_tpl = bool(getattr(tok, "chat_template", None))
    if not has_chat_tpl and not args.fallback_prompt:
        print(f"[hf] WARNING: {args.model} has no chat template; "
              f"passing system+user concatenated. Use --fallback_prompt to silence, "
              f"or --chat_template llama3|mistral to inject one.")

    def _fold_system_into_user(msgs):
        """For chat templates that disallow `system` role (e.g. Mistral),
        prepend the system content to the first user message."""
        sys_parts = [m["content"] for m in msgs if m.get("role") == "system"]
        non_sys = [m for m in msgs if m.get("role") != "system"]
        if not sys_parts or not non_sys:
            return msgs
        sys_text = "\n\n".join(sys_parts)
        merged = [dict(m) for m in non_sys]
        first_user_idx = next(
            (i for i, m in enumerate(merged) if m.get("role") == "user"), 0,
        )
        merged[first_user_idx]["content"] = (
            sys_text + "\n\n" + (merged[first_user_idx].get("content") or "")
        )
        return merged

    def _render(messages):
        if not has_chat_tpl:
            sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            usr = next((m["content"] for m in messages if m["role"] == "user"), "")
            return sys_msg + "\n\n" + usr + "\n\nAssistant:"
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception as e_tpl:
            # Mistral-family templates reject `system` role -> fold it into user[0]
            folded = _fold_system_into_user(messages)
            return tok.apply_chat_template(
                folded, tokenize=False, add_generation_prompt=True,
            )

    n_ok = n_fail = 0
    t0 = time.time()
    with out_path.open("a") as fout:
        for i, pid in enumerate(todo, 1):
            messages = prompts[pid]
            try:
                text = _render(messages)
                enc = tok([text], return_tensors="pt", padding=True,
                          truncation=True, max_length=8192)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                input_len = enc["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                raw = tok.decode(out[0, input_len:], skip_special_tokens=True)
            except Exception as e:
                print(f"[hf] {pid} generation failed: {e}", file=sys.stderr)
                continue
            trace = parse_trace(raw)
            final = trace_to_final_prediction(trace)
            rec = {
                "pair_id": pid,
                "input_order": args.input_order,
                "model": args.model,
                "final_prediction": final,
                "raw_trace_present": trace is not None,
            }
            if trace is None:
                rec["raw_output"] = (raw or "")[:4000]
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if trace is None:
                n_fail += 1
            else:
                n_ok += 1
            if i % 10 == 0:
                rate = i / max(time.time() - t0, 1.0)
                print(f"[hf] {i}/{len(todo)} parse_ok={n_ok} parse_fail={n_fail} ({rate:.2f} rec/s)")

    print(f"[hf] done. parse_ok={n_ok} fail={n_fail}; wrote {out_path}")


if __name__ == "__main__":
    main()
