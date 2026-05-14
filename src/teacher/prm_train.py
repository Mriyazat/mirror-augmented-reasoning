"""B3 — DDI-PRM fine-tune script.  Runs on 4×H100.

Starts from `dmis-lab/llama-3.1-medprm-reward-v1.0` (8B Llama-3.1-Instruct
that the Med-PRM team already fine-tuned into a step-level PRM on medical
reasoning; EMNLP '25 artifact) and LoRA-fine-tunes it on our DDI traces
so it learns DDI-specific failure modes (evidence grounding, direction,
family, summary).  This shortens convergence vs. training from a plain
instruct model, per the Med-PRM EMNLP '25 finding.

Alternative base (flag `--base_model`):
  - `dmis-lab/llama-3.1-medprm-reward-v1.0`   (default — recommended; already PRM)
  - `meta-llama/Llama-3.1-8B-Instruct`        (fallback; train from scratch)
  - `Qwen/Qwen2.5-7B-Instruct`                (out-of-family ablation)

All paths use the same recipe:  classification-as-language-modeling — for
each ` ки` position the next-token supervision is ' +' or ' -'.  We only
compute loss on those target tokens; the rest is masked out.

Launch on the cluster:

    source scripts/slurm/activate_env.sh
    accelerate launch --config_file configs/accelerate_fsdp.yaml \\
        src/teacher/prm_train.py \\
        --base_model dmis-lab/llama-3.1-medprm-reward-v1.0 \\
        --train_file outputs/teacher/prm_train.jsonl \\
        --eval_file  outputs/teacher/prm_eval.jsonl \\
        --output_dir $DDI_CKPT/ddi_prm_v1 \\
        --epochs 2 --lr 1e-4 --batch_size 2 --grad_accum 8

The script is fully self-contained — no cluster-side edits needed unless
you change the base model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Lazy-imported; these fail gracefully if not on a GPU node
try:
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer,
                              DataCollatorForSeq2Seq)
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    _GPU_OK = True
except ImportError as e:
    _GPU_OK = False
    _IMPORT_ERR = e


SEP_TOKEN = " ки"
POS_TOKEN = " +"
NEG_TOKEN = " -"

SYSTEM_PROMPT = (
    "You are an evaluator assessing the logicality and validity of each step "
    "of the following DDI reasoning trace.  For each reasoning step, output + "
    "if the step is logically valid and evidence-grounded; output - if the "
    "step contains an error (hallucinated IDs, flipped direction, off-family "
    "family_hint, silent abstention).  In addition, the question block "
    "contains the query pair and supporting evidence context."
)


def build_example_text(question: str, solution: str) -> str:
    """Pre-target input — exactly the text the PRM will see at inference."""
    return (
        f"[System]\n{SYSTEM_PROMPT}\n\n"
        f"[Question]\n{question}\n\n"
        f"[Solution]\n{solution}"
    )


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def tokenize_with_step_targets(example: dict, tokenizer) -> dict:
    """Turns one QC example into input_ids + labels where labels are -100
    everywhere except at the position right after each ` ки`, which is set
    to the ID of ' +' or ' -' depending on step_labels.

    This is Med-PRM's supervision format (see their scripts/2_training.sh).
    """
    text = build_example_text(example["question"], example["solution"])
    enc = tokenizer(text, return_offsets_mapping=True,
                    add_special_tokens=True, return_tensors="pt",
                    truncation=True, max_length=3072)
    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0]

    # Find positions of " ки" in the text; the TOKEN AFTER each of them is
    # where we want +/- supervision.
    sep_positions_text = []
    i = 0
    while True:
        j = text.find(SEP_TOKEN, i)
        if j < 0:
            break
        sep_positions_text.append(j)
        i = j + len(SEP_TOKEN)

    # Map text positions to token indices
    token_sep_idx = []
    for pos in sep_positions_text:
        for ti, (s, e) in enumerate(offsets.tolist()):
            if s <= pos < e:
                token_sep_idx.append(ti)
                break

    # Compute plus_id / minus_id. Some tokenizers need the leading space.
    plus_id = tokenizer(POS_TOKEN, add_special_tokens=False)["input_ids"][0]
    minus_id = tokenizer(NEG_TOKEN, add_special_tokens=False)["input_ids"][0]

    # Labels: all -100 except right after each ` ки`
    labels = torch.full_like(input_ids, fill_value=-100)
    step_labels = example["step_labels"]
    # Be defensive: if we found more separators than labels (truncation), cap
    n_sep = min(len(token_sep_idx), len(step_labels))
    for i in range(n_sep):
        tgt_token_idx = token_sep_idx[i] + 1     # position to predict AFTER ки
        if tgt_token_idx >= len(input_ids):
            continue
        labels[tgt_token_idx] = plus_id if step_labels[i] else minus_id

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": enc["attention_mask"][0].tolist(),
        "labels": labels.tolist(),
    }


def main():
    if not _GPU_OK:
        raise RuntimeError(f"GPU deps not available: {_IMPORT_ERR}")

    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="dmis-lab/llama-3.1-medprm-reward-v1.0",
                   help="PRM base model (dmis-lab/llama-3.1-medprm-reward-v1.0 "
                        "recommended — EMNLP '25 Med-PRM reward checkpoint; use "
                        "meta-llama/Llama-3.1-8B-Instruct to reproduce the "
                        "Med-PRM paper recipe from scratch)")
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_len", type=int, default=3072)
    p.add_argument("--resume", action="store_true",
                   help="Auto-resume from the latest checkpoint in --output_dir "
                        "if any exist; otherwise start fresh. Safe to set on "
                        "every relaunch.")
    p.add_argument("--eval_steps", type=int, default=1000,
                   help="Run eval every N train steps. Eval on the full 89k "
                        "set takes ~30 min, so don't set this too aggressive.")
    p.add_argument("--save_steps", type=int, default=200,
                   help="Save a checkpoint every N train steps. Keep this "
                        "small for resume safety on time-limited allocations.")
    p.add_argument("--eval_batch_size", type=int, default=16,
                   help="Per-device eval batch size. Eval has no backward "
                        "pass so this can be 4-8x larger than train.")
    p.add_argument("--max_eval_examples", type=int, default=2000,
                   help="Cap eval set size for in-training monitoring. The "
                        "full eval set is too expensive to run every cycle; "
                        "a 2k sample is enough to detect divergence. Set to "
                        "0 to use the full eval set (slow).")
    args = p.parse_args()

    print(f"[prm-train] loading {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prefer flash_attention_2 when available (H100 likes it), but fall back
    # to SDPA when the wheel isn't installed in the venv. SDPA is ~95% of
    # flash-attn's throughput at these seq lengths and ships with PyTorch
    # so it never requires extra cluster setup.
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"
        print(f"[prm-train] flash_attn not available; using attn_implementation={attn_impl!r}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    # Disable KV cache for training: keeping it on with SDPA backward + GQA
    # corrupts shapes during gradient-checkpoint recompute, and even without
    # checkpointing it wastes memory.
    model.config.use_cache = False
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    print("[prm-train] tokenizing ...", flush=True)
    train = load_jsonl(Path(args.train_file))
    eval_ = load_jsonl(Path(args.eval_file))
    if args.max_eval_examples and len(eval_) > args.max_eval_examples:
        import random as _r
        _r.Random(0).shuffle(eval_)
        eval_full = len(eval_)
        eval_ = eval_[: args.max_eval_examples]
        print(f"[prm-train] capping eval set: {eval_full:,} -> "
              f"{len(eval_):,} (use --max_eval_examples 0 for the full set)",
              flush=True)
    train_ds = Dataset.from_list(
        [tokenize_with_step_targets(x, tokenizer) for x in train])
    eval_ds = Dataset.from_list(
        [tokenize_with_step_targets(x, tokenizer) for x in eval_])
    print(f"[prm-train] train={len(train_ds):,}  eval={len(eval_ds):,}")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8,
                                      return_tensors="pt")
    trainer = Trainer(model=model, args=targs,
                      train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=collator,
                      tokenizer=tokenizer)

    resume_arg = False
    if args.resume:
        ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1])
                       if p.name.split("-")[1].isdigit() else -1)
        if ckpts:
            resume_arg = str(ckpts[-1])
            print(f"[prm-train] resuming from {resume_arg}", flush=True)
        else:
            print(f"[prm-train] --resume set but no checkpoints in "
                  f"{args.output_dir}; starting fresh", flush=True)

    trainer.train(resume_from_checkpoint=resume_arg if resume_arg else None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[prm-train] done — saved to {args.output_dir}")


if __name__ == "__main__":
    main()
