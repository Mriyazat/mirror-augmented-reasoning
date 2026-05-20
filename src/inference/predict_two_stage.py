"""Phase 3.5 / 5.4 -- Two-stage inference: classifier head + LoRA trace.

Stage 1 (head). For each record we forward the prompt through the LoRA-
adapted model with `output_hidden_states=True`, take the last-position
last-layer hidden state, standardise it with the head's saved mean/std,
and apply $W h + b$ to get a 7-way family logit vector.  The head's
arg-max is the family decision.

Stage 2 (LoRA trace). We then run the usual generation through the
LoRA adapter to produce the JSON trace with subtype, direction tag and
summary.  The LoRA's family is **overridden** by the head's family when
they disagree (rationale: the head was trained on the imbalanced family
distribution with class-balanced loss, the LoRA was not).  We also
record both predictions so the final agreement rate can be analysed.

Output schema
-------------
Same as `src/inference/predict.py`, with two additions:
    {
      ... (same as predict.py output) ...,
      "head_prediction": {
        "family":       str,
        "label_dist":   { family: prob, ... },
        "agree_lora":   bool,
      },
      "final_prediction": {
        "family":          <head's family>,
        "lora_family":     <what LoRA emitted>,
        "subtype":         <LoRA's subtype>,
        "direction_tag":   <LoRA's direction tag>,
        ...
      }
    }

Usage
-----
    python -m src.inference.predict_two_stage \
        --adapter   $DDI_CKPT/student/ddi_v4_best_phase4_prm_dpo_macro0797 \
        --head      outputs/student/ddi_v4_phase4/head/head.npz \
        --input     outputs/eval_prompts/random_full_test_5000_stratified.with_neighbors.prompts.jsonl \
        --output    outputs/eval_prompts/pred_${RUN}_2stage_ab.jsonl \
        --batch     4 \
        --max_new_tokens 768
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
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


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _load_head(head_path: str | Path) -> dict:
    d = np.load(head_path, allow_pickle=True)
    families = list(d["families"])
    head = {
        "W":          d["W"].astype(np.float32),
        "b":          d["b"].astype(np.float32),
        "mean":       d["mean"].astype(np.float32),
        "std":        d["std"].astype(np.float32),
        "families":   families,
        "hidden_dim": int(d["hidden_dim"]),
    }
    print(f"[2stage] loaded head: hidden_dim={head['hidden_dim']}, "
          f"families={families}")
    return head


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
            base_model, dtype=dtype, trust_remote_code=True,
            device_map=device_map,
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, trust_remote_code=True,
            device_map=device_map,
        )
    model = PeftModel.from_pretrained(base, adapter_dir) if adapter_dir else base
    model.eval()
    return model, tokenizer


def _head_predict_batch(model, tokenizer, head: dict, prompts: list[str],
                        max_length: int) -> tuple[list[str], list[dict]]:
    """Run a forward pass on the prompts (no generation), pull the last-
    position last-layer hidden state, and apply the head.  Returns
    (predicted_family_per_record, label_dist_per_record).

    label_dist is {family_name: prob}.  Sums to 1.
    """
    enc = tokenizer(
        prompts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
        add_special_tokens=False,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    last_hidden = out.hidden_states[-1]      # [B, T, H]
    attn = enc["attention_mask"]
    last_idx = attn.sum(dim=1) - 1           # [B]
    arange = torch.arange(last_hidden.size(0), device=model.device)
    pooled = last_hidden[arange, last_idx, :].float().cpu().numpy()   # [B, H]

    # standardise + linear head
    X = (pooled - head["mean"]) / np.maximum(head["std"], 1e-8)
    logits = X @ head["W"].T + head["b"]
    probs = _softmax_np(logits, axis=-1)
    pred_idx = probs.argmax(axis=-1)

    fams = [head["families"][int(i)] for i in pred_idx]
    dists: list[dict] = []
    for p in probs:
        dists.append({head["families"][i]: float(p[i])
                      for i in range(len(head["families"]))})
    return fams, dists


def _generate_batch(model, tokenizer, prompts: list[str],
                    max_new_tokens: int) -> list[str]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=4096)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    input_len = enc["input_ids"].shape[1]
    gen = out[:, input_len:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--head", required=True,
                   help="Path to head.npz from train_classifier_head.py")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--max_length", type=int, default=4096,
                   help="Max prompt length for the head forward.")
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device_map", default="auto")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--input_order", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--with_context", dest="with_context", action="store_true")
    p.add_argument("--no_context", dest="with_context", action="store_false")
    p.set_defaults(with_context=True)
    p.add_argument("--head_overrides_lora", action="store_true",
                   help="Default behaviour. The head's family wins when it "
                        "disagrees with the LoRA's family.")
    p.add_argument("--lora_overrides_head", dest="head_overrides_lora",
                   action="store_false",
                   help="Use the LoRA's family even when it disagrees with "
                        "the head. For ablation only.")
    p.set_defaults(head_overrides_lora=True)
    args = p.parse_args()

    input_order = args.input_order or ("ba" if args.mirror else "ab")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[2stage] loading model from {args.adapter}")
    model, tokenizer = _load_model(
        args.base_model, args.adapter, args.torch_dtype, args.device_map,
    )
    head = _load_head(args.head)
    if int(model.config.hidden_size) != head["hidden_dim"]:
        raise SystemExit(
            f"head hidden_dim={head['hidden_dim']} != "
            f"model hidden_size={model.config.hidden_size}; "
            "head was trained against a different backbone."
        )

    records = []
    with open(args.input) as fin:
        for line in fin:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit > 0:
        records = records[:args.limit]

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
            print(f"[2stage] --resume: kept {len(done_ids)} prior records, "
                  f"{len(records)} still to predict.")
        except Exception as e:
            print(f"[2stage] --resume: failed to read {out_path} ({e}); overwriting.")

    print(f"[2stage] {len(records)} input records, "
          f"batch={args.batch}, mirror={args.mirror}, "
          f"head_overrides_lora={args.head_overrides_lora}")

    n_parse_ok = 0
    n_parse_fail = 0
    n_agree = 0
    n_disagree = 0
    t0 = time.time()
    with out_path.open(open_mode) as fout:
        for i in range(0, len(records), args.batch):
            chunk = records[i:i + args.batch]
            prompts = []
            for rec in chunk:
                msgs = _build_prompt_messages(rec, args.mirror)
                prompts.append(
                    tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                )
            head_fams, head_dists = _head_predict_batch(
                model, tokenizer, head, prompts, args.max_length,
            )
            gens = _generate_batch(
                model, tokenizer, prompts, args.max_new_tokens,
            )

            for rec, raw, head_fam, head_dist in zip(chunk, gens, head_fams, head_dists):
                parsed, err = _extract_json(raw)
                if parsed is None:
                    parsed = {}
                    err = err or "no_json"
                    n_parse_fail += 1
                else:
                    n_parse_ok += 1
                shaped = _normalize_trace(parsed, rec)
                lora_fam = shaped["final_prediction"].get("family")
                agree = (lora_fam == head_fam)
                if agree:
                    n_agree += 1
                else:
                    n_disagree += 1

                final_fam = head_fam if args.head_overrides_lora else lora_fam
                final_pred = dict(shaped["final_prediction"])
                final_pred["family"]      = final_fam
                final_pred["lora_family"] = lora_fam
                final_pred["label_dist"]  = head_dist

                pair_id = rec.get("pair_id", "")
                ctx_ids = rec.get("context_ids") or []
                if args.with_context and not ctx_ids:
                    ctx_ids = _get_context_ids(pair_id)

                out_rec = {
                    "pair_id":          pair_id,
                    "input_order":      input_order,
                    "context_ids":      ctx_ids,
                    "trace":            shaped["trace"],
                    "final_prediction": final_pred,
                    "head_prediction": {
                        "family":     head_fam,
                        "label_dist": head_dist,
                        "agree_lora": agree,
                    },
                    "raw_output": raw,
                    "parse_ok":   err is None,
                    "parse_error": err,
                }
                fout.write(json.dumps(out_rec) + "\n")

            done = min(i + args.batch, len(records))
            if done % (args.batch * 20) == 0 or done == len(records):
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                print(f"[2stage] {done}/{len(records)} "
                      f"parse_ok={n_parse_ok} fail={n_parse_fail} "
                      f"agree={n_agree} disagree={n_disagree} "
                      f"({rate:.2f} rec/s)")

    print(f"[2stage] done. parse_ok={n_parse_ok} parse_fail={n_parse_fail} "
          f"head_lora_agree={n_agree} disagree={n_disagree} "
          f"({100*n_agree/max(n_agree+n_disagree,1):.1f}% agreement) "
          f"out={out_path}")
    if args.with_context:
        print(f"[2stage] context_ids stats: {_CTX_STATS}")
    if args.mirror:
        print(f"[2stage] mirror stats: {_MIRROR_STATS}")


if __name__ == "__main__":
    main()
