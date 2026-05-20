"""Mirror-DPO / IPO with PRM-weighted preferences.

Background
----------
Distilled DDI students exhibit a characteristic failure mode in which a
large fraction of their errors are attributable to copying the
direction_tag across the (A, B) and (B, A) orderings instead of flipping
it. SFT alone does not fully resolve this -- the symmetry-KL loss in
`sft_train.py` reduces the gap but residual mirror errors remain.

The fix: **Mirror-DPO** (or IPO, which is more robust to noisy
preferences).  We construct preference pairs where

    chosen   = teacher trace that got the direction right
    rejected = counterfactual trace with the WRONG direction tag
               (plus matching changes to the conclusion)

and train the student to prefer the chosen version.

Why IPO (Azar et al., 2024) over DPO
------------------------------------
Mirror preferences are synthetic -- we generate the "rejected" trace
by perturbing the "chosen" trace.  Synthetic preferences can be
noisy (some rejected variants are accidentally plausible).  DPO's
log-sigmoid loss saturates on ambiguous pairs and can chase them,
collapsing the policy.  IPO's squared-hinge loss

    L_IPO = E[(h_theta - 1 / (2 beta))^2]      h_theta = log ratio diff

is bounded and handles noisy labels gracefully.  We default to IPO.

PRM weighting
-------------
When DDI-PRM scores are available on the chosen trace's steps, we
weight each preference pair by the *mean PRM score over steps* of the
chosen trace.  Intuition: a well-supported chosen trace is a strong
teaching example; a weakly-supported one (noise) should contribute
less.  This is the knob that makes Mirror-DPO *interpretable*: we can
ablate "-- PRM weighting" and measure the delta.

Preference-pair format (JSONL)
------------------------------
Each line:
    {
      "pair_id":     str,
      "mirror_type": "direction_flip" | "family_swap" | "evidence_drop"
                     | "abstain_unsafe" | ...,
      "prompt":      [chat messages up through the final user turn],
      "chosen":      "<full assistant reply as string>",
      "rejected":    "<full assistant reply as string>",
      "prm_chosen":  float | null,      # mean PRM over chosen's steps
      "prm_rejected": float | null,      # (optional; diagnostic only)
    }

The accompanying `src/teacher/build_preference_pairs.py` script
(run after teacher generation completes) emits this file.  This
trainer consumes it directly.

Why not use TRL's DPOTrainer as-is?
-----------------------------------
TRL's `DPOTrainer` is a solid base but
  (a) doesn't support per-pair loss weighting (we need this for PRM),
  (b) its IPO support became less flexible in newer versions,
  (c) we want to optionally bypass a reference model in
      extreme-memory settings by snapshotting the base-LoRA-disabled
      model inline.

So we subclass `DPOTrainer` and override `get_batch_loss_metrics` to
add PRM weighting.  If TRL isn't available we fall back to an
in-module `_ManualDpoTrainer` that implements the core loss without
TRL.  Either way the CLI is identical.

Usage
-----
    python -m src.training.dpo_mirror \
        --sft_adapter  outputs/student/ddi_v4_sft/checkpoint-best \
        --pref_file    outputs/preferences/mirror_pairs.jsonl \
        --output_dir   outputs/student/ddi_v4_dpo \
        --loss_type    ipo \
        --beta         0.1 \
        --epochs       1 \
        --use_prm_weight
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from src.teacher.evidence_resolution import expand_context_ids  # noqa: F401

# --- soft imports ----------------------------------------------------------
try:
    import torch                                                    # noqa: F401
    import torch.nn.functional as F                                 # noqa: F401
    from torch.utils.data import Dataset                            # type: ignore
    _HAVE_TORCH = True
except Exception as _torch_err:
    _HAVE_TORCH = False
    _TORCH_IMPORT_ERR = _torch_err

    class Dataset:  # type: ignore[no-redef]
        """Placeholder so class definitions still parse without torch."""

try:
    from transformers import (                                      # noqa: E402
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    )
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType  # noqa: E402
    _HAVE_HF = True
except Exception as _hf_err:
    _HAVE_HF = False
    _HF_IMPORT_ERR = _hf_err

try:
    from trl import DPOTrainer, DPOConfig                            # noqa: E402
    _HAVE_TRL = True
except Exception as _trl_err:
    _HAVE_TRL = False
    _TRL_IMPORT_ERR = _trl_err

try:
    from datasets import Dataset as HfDataset                         # noqa: E402
    _HAVE_DATASETS = True
except Exception as _datasets_err:
    _HAVE_DATASETS = False
    _DATASETS_IMPORT_ERR = _datasets_err


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ====================================================================
# Dataset
# ====================================================================
class PreferenceJsonlDataset(Dataset):
    """One preference pair per record.

    Each record must provide:
        prompt      : list[dict]   chat messages up through final user turn
        chosen      : str          assistant reply (preferred)
        rejected    : str          assistant reply (dispreferred)
        prm_chosen  : float | None optional PRM mean over chosen's steps

    This class does the minimal thing TRL expects from a preference
    dataset: dict-style access with keys "prompt", "chosen", "rejected".
    Per-pair weights are attached in a side-channel consumed by the
    custom trainer.
    """

    def __init__(self, path: str | Path, tokenizer=None,
                 prm_weight_clamp: tuple[float, float] = (0.25, 1.0),
                 prm_weight_transform: str = "linear"):
        self.records: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.prm_weight_clamp = prm_weight_clamp
        self.prm_weight_transform = prm_weight_transform

    def __len__(self) -> int:
        return len(self.records)

    def _prm_weight(self, rec: dict) -> float:
        """Map PRM mean-score -> per-pair weight.

        Default "linear": rescale [0, 1] -> [clamp_lo, clamp_hi].
        Alternative "sigmoid": sharper cutoff at 0.5.
        """
        prm = rec.get("prm_chosen")
        if prm is None:
            return 1.0
        lo, hi = self.prm_weight_clamp
        if self.prm_weight_transform == "linear":
            w = lo + (hi - lo) * float(prm)
        elif self.prm_weight_transform == "sigmoid":
            import math
            w = lo + (hi - lo) * (1.0 / (1.0 + math.exp(-10.0 * (prm - 0.5))))
        else:
            w = float(prm)
        return max(lo, min(hi, w))

    def _prompt_string(self, rec: dict) -> str:
        """Render prompt messages via the tokenizer's chat template."""
        if self.tokenizer is None:
            return json.dumps(rec["prompt"])
        return self.tokenizer.apply_chat_template(
            rec["prompt"], tokenize=False, add_generation_prompt=True,
        )

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        prompt_str = self._prompt_string(rec)
        return {
            # TRL DPOTrainer expects these keys verbatim:
            "prompt":   prompt_str,
            "chosen":   rec["chosen"],
            "rejected": rec["rejected"],
            # Side channels:
            "pair_id":      rec.get("pair_id", ""),
            "mirror_type":  rec.get("mirror_type", "unknown"),
            "prm_weight":   self._prm_weight(rec),
        }

    def to_list(self) -> list[dict]:
        """Materialize records in the schema expected by TRL DPOTrainer."""
        return [self[i] for i in range(len(self))]


def _as_hf_dataset(ds: PreferenceJsonlDataset, *,
                   weighted_sampling: bool = False,
                   seed: int = 42,
                   label: str = "train"):
    """Convert our lightweight JSONL wrapper into HF datasets.Dataset.

    Newer TRL versions call `.map()` during DPOTrainer initialization, so a
    plain PyTorch Dataset is no longer accepted. We keep the JSONL wrapper for
    local/manual fallback, but feed TRL a real HuggingFace Dataset.
    """
    if ds is None:
        return None
    if not _HAVE_DATASETS:
        raise RuntimeError(
            "TRL DPOTrainer requires `datasets.Dataset` for this installed "
            f"TRL version, but `datasets` is not importable: {_DATASETS_IMPORT_ERR!r}"
        )
    rows = ds.to_list()
    if weighted_sampling:
        rows = _importance_sample_by_prm_weight(rows, seed=seed, label=label)
    return HfDataset.from_list(rows)


def _importance_sample_by_prm_weight(rows: list[dict], *,
                                     seed: int,
                                     label: str) -> list[dict]:
    """Fallback PRM weighting for TRL versions without per-example losses.

    Some TRL releases no longer expose a batch-loss hook we can wrap. In that
    case we still make PRM quality affect optimization by deterministic
    importance sampling: examples with lower `prm_weight` are retained less
    often during training. This is not identical to per-loss weighting, but it
    is far better than silently ignoring the PRM signal and is compatible with
    TRL's public Dataset interface.
    """
    if not rows:
        return rows
    weights = [float(r.get("prm_weight", 1.0) or 1.0) for r in rows]
    max_w = max(max(weights), 1e-6)
    rng = random.Random(seed)
    kept: list[dict] = []
    for row, weight in zip(rows, weights):
        keep_prob = max(0.0, min(1.0, weight / max_w))
        if rng.random() <= keep_prob:
            kept.append(row)
    if not kept:
        kept.append(rows[weights.index(max(weights))])
    print(
        f"[dpo] PRM-weight fallback via {label} importance sampling: "
        f"kept {len(kept):,}/{len(rows):,} examples "
        f"(mean_weight={sum(weights) / len(weights):.3f}, max_weight={max_w:.3f})"
    )
    return kept


# ====================================================================
# TRL-based trainer (preferred path)
# ====================================================================
def _build_trl_trainer(model, ref_model, tokenizer, train_ds, val_ds,
                       args: argparse.Namespace) -> "DPOTrainer":
    """Construct and customize the TRL DPOTrainer.

    We override the loss to inject per-pair PRM weights.  TRL's
    default loss computes:
        losses = loss_type(policy_chosen_logps, policy_rejected_logps,
                           ref_chosen_logps, ref_rejected_logps)
    We intercept and multiply element-wise by per-pair weights before
    the batch-mean reduction.
    """
    assert _HAVE_TRL, "TRL not available; run with --no_trl to fall back."

    # TRL's DPOConfig API has changed across versions. Build a superset of
    # useful kwargs and pass only those accepted by the installed version so a
    # cluster-side package update/downgrade doesn't crash before training.
    dpo_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_per_gpu,
        per_device_eval_batch_size=args.batch_per_gpu,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        bf16=(args.torch_dtype == "bfloat16"),
        fp16=(args.torch_dtype == "float16"),
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[],
        remove_unused_columns=False,   # we need the side-channel columns
        loss_type=args.loss_type,       # "sigmoid" (DPO) | "ipo" | "hinge" | ...
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    dpo_params = inspect.signature(DPOConfig.__init__).parameters
    accepted = {k: v for k, v in dpo_kwargs.items() if k in dpo_params}
    skipped = sorted(set(dpo_kwargs) - set(accepted))
    if skipped:
        print(f"[dpo] DPOConfig does not accept {skipped}; skipping them")
    dpo_config = DPOConfig(**accepted)

    prm_patch_mode = "none"
    if args.use_prm_weight:
        prm_patch_mode = _detect_prm_patch_mode()
        if prm_patch_mode == "unsupported":
            if args.prm_weight_fallback == "error":
                raise RuntimeError(
                    "Requested --use_prm_weight, but this TRL DPOTrainer does "
                    "not expose a supported per-loss hook. Use "
                    "--prm_weight_fallback sample to run deterministic "
                    "importance sampling, or --no_trl to use the manual "
                    "weighted trainer for a small smoke run."
                )
            print("[dpo] TRL per-loss PRM weighting is unavailable; using "
                  "deterministic PRM-weighted importance sampling fallback")
    train_hf = _as_hf_dataset(
        train_ds,
        weighted_sampling=(args.use_prm_weight and prm_patch_mode == "unsupported"
                           and args.prm_weight_fallback == "sample"),
        seed=args.seed,
        label="train",
    )
    val_hf = _as_hf_dataset(val_ds) if val_ds is not None else None

    trainer_kwargs = {
        "model": model,
        "ref_model": ref_model,
        "args": dpo_config,
        "train_dataset": train_hf,
        "eval_dataset": val_hf,
    }
    dpo_init_params = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in dpo_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)

    if args.use_prm_weight and prm_patch_mode == "dpo_loss_hook":
        _patch_trainer_with_prm_weight(trainer)

    return trainer


def _detect_prm_patch_mode() -> str:
    if hasattr(DPOTrainer, "get_batch_loss_metrics") and hasattr(DPOTrainer, "dpo_loss"):
        return "dpo_loss_hook"
    return "unsupported"


def _patch_trainer_with_prm_weight(trainer: "DPOTrainer") -> None:
    """Monkey-patch the trainer's loss computation to inject PRM weights.

    Recent TRL versions compute a per-example preference-loss vector inside
    `dpo_loss(...)`, then `get_batch_loss_metrics(...)` averages it. We wrap
    both methods:

    1. `get_batch_loss_metrics` captures the current batch's `prm_weight`.
    2. `dpo_loss` multiplies the per-example loss vector by normalized weights
       before TRL's own reduction.

    This is exact per-loss PRM weighting, not preference sampling.
    """
    if not hasattr(trainer, "get_batch_loss_metrics") or not hasattr(trainer, "dpo_loss"):
        raise RuntimeError(
            "Installed TRL DPOTrainer lacks get_batch_loss_metrics/dpo_loss; "
            "cannot apply exact per-example PRM weighting."
        )
    orig_get_batch_loss_metrics = trainer.get_batch_loss_metrics
    orig_dpo_loss = trainer.dpo_loss

    def wrapped_get_batch_loss_metrics(model, batch, train_eval="train"):
        weights = batch.get("prm_weight")
        trainer._ddi_prm_weight = weights
        try:
            return orig_get_batch_loss_metrics(model, batch, train_eval=train_eval)
        finally:
            trainer._ddi_prm_weight = None

    def wrapped_dpo_loss(*args, **kwargs):
        result = orig_dpo_loss(*args, **kwargs)
        if not isinstance(result, tuple) or not result:
            return result
        losses = result[0]
        weights = getattr(trainer, "_ddi_prm_weight", None)
        if weights is not None and _HAVE_TORCH and torch.is_tensor(losses) and losses.ndim > 0:
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=losses.dtype, device=losses.device)
            weights = weights.to(losses.device).to(losses.dtype)
            if weights.numel() == losses.numel():
                w_norm = weights.mean().clamp(min=1e-6)
                losses = losses * (weights / w_norm)
            else:
                raise RuntimeError(
                    f"PRM weight shape mismatch: weights={tuple(weights.shape)} "
                    f"losses={tuple(losses.shape)}"
                )
        return (losses, *result[1:])

    trainer.get_batch_loss_metrics = wrapped_get_batch_loss_metrics
    trainer.dpo_loss = wrapped_dpo_loss
    print("[dpo] exact per-loss PRM weighting enabled via TRL dpo_loss hook")


# ====================================================================
# Manual fallback trainer (no TRL)
# ====================================================================
class _ManualDpoTrainer:
    """Minimal DPO/IPO implementation for environments without TRL.

    Not intended to match TRL's throughput or features (no FlashAttn
    batching tricks, no length balancing, no eval hooks).  It's here
    so CI + local smoke tests still work.  Production runs MUST use
    the TRL path.
    """

    def __init__(self, model, ref_model, tokenizer, train_ds, val_ds,
                 args: argparse.Namespace):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.args = args
        if not _HAVE_TORCH:
            raise RuntimeError(
                "torch unavailable; cannot run _ManualDpoTrainer. "
                f"import error: {_TORCH_IMPORT_ERR}"
            )

    def _logprob_of(self, model, prompt_str: str, reply: str) -> "torch.Tensor":
        full = prompt_str + reply
        enc_p = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        enc_f = self.tokenizer(full,       return_tensors="pt", add_special_tokens=False)
        # Labels: -100 for prompt, reply ids for reply.
        labels = enc_f["input_ids"].clone()
        labels[:, : enc_p["input_ids"].shape[1]] = -100
        out = model(enc_f["input_ids"].to(model.device), labels=labels.to(model.device))
        # Negative because HF returns a *mean* NLL per non-ignored token;
        # we want sum log-prob, not mean.
        n_tok = (labels != -100).sum().clamp(min=1)
        return -out.loss * n_tok

    def _batch_loss(self, batch: list[dict]) -> "torch.Tensor":
        if not _HAVE_TORCH:
            raise RuntimeError("torch unavailable")
        import torch  # local alias for clarity
        pi_cw  = torch.stack([self._logprob_of(self.model, b["prompt"], b["chosen"])   for b in batch])
        pi_rw  = torch.stack([self._logprob_of(self.model, b["prompt"], b["rejected"]) for b in batch])
        with torch.no_grad():
            ref_cw = torch.stack([self._logprob_of(self.ref_model, b["prompt"], b["chosen"])   for b in batch])
            ref_rw = torch.stack([self._logprob_of(self.ref_model, b["prompt"], b["rejected"]) for b in batch])

        # logratio = log pi(y_w)/pi_ref(y_w) - log pi(y_l)/pi_ref(y_l)
        h = (pi_cw - ref_cw) - (pi_rw - ref_rw)

        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.args.beta * h)
        elif self.args.loss_type == "ipo":
            target = 1.0 / (2.0 * self.args.beta)
            losses = (h - target) ** 2
        elif self.args.loss_type == "hinge":
            losses = F.relu(1.0 - self.args.beta * h)
        else:
            raise ValueError(f"unknown loss_type {self.args.loss_type!r}")

        if self.args.use_prm_weight:
            weights = torch.tensor([b["prm_weight"] for b in batch],
                                   dtype=losses.dtype, device=losses.device)
            w_norm = weights.mean().clamp(min=1e-6)
            losses = losses * (weights / w_norm)

        return losses.mean()

    def train(self) -> None:
        import torch
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        bs = self.args.batch_per_gpu
        n = len(self.train_ds)
        steps_per_epoch = max(1, n // bs)

        for epoch in range(self.args.epochs):
            running = 0.0
            for step in range(steps_per_epoch):
                batch = [self.train_ds[i] for i in range(step * bs,
                                                          min(n, (step + 1) * bs))]
                if not batch:
                    break
                loss = self._batch_loss(batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item()
                if step % self.args.logging_steps == 0:
                    print(f"[manual-dpo] epoch={epoch} step={step} "
                          f"loss={running / max(1, step + 1):.4f}")

            ck = Path(self.args.output_dir) / f"checkpoint-epoch{epoch}"
            ck.mkdir(parents=True, exist_ok=True)
            try:
                # Save LoRA adapter only if PEFT wrapped
                self.model.save_pretrained(str(ck))
            except Exception:
                torch.save(self.model.state_dict(), ck / "state_dict.pt")
            self.tokenizer.save_pretrained(str(ck))


# ====================================================================
# Model building
# ====================================================================
def _build_model_and_ref(args: argparse.Namespace):
    if not _HAVE_HF:
        raise RuntimeError(
            f"transformers/peft unavailable: {_HF_IMPORT_ERR}"
        )

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True,
                                        padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    # Policy: base model + SFT adapter attached.
    try:
        base = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    if args.sft_adapter:
        policy = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)
    else:
        # No SFT adapter: attach a fresh LoRA for DPO-only (rare).
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none", task_type=TaskType.CAUSAL_LM,
        )
        policy = get_peft_model(base, lora_cfg)
    # PEFT adapters may load in fp32 even when the base model is bf16. FSDP
    # requires uniform dtype inside flattened parameter groups, so normalize the
    # wrapped policy before Accelerate/TRL prepares it.
    policy = policy.to(dtype=dtype)

    policy.config.use_cache = False
    if args.gradient_checkpointing:
        if hasattr(policy, "enable_input_require_grads"):
            policy.enable_input_require_grads()
        else:
            def _make_inputs_require_grad(_module, _inputs, output):
                output.requires_grad_(True)
            policy.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

    # Reference: base model WITH SFT adapter, frozen.  This is crucial
    # -- DPO compares the DPO-policy against the SFT-initialized
    # policy, not the raw pretrained one, so the only "preference
    # learning" delta is on mirror data.
    try:
        ref_base = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype)
    except TypeError:
        ref_base = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    if args.sft_adapter:
        ref_model = PeftModel.from_pretrained(ref_base, args.sft_adapter,
                                              is_trainable=False)
    else:
        ref_model = ref_base
    # Same dtype normalization for the frozen reference. The cluster TRL path
    # wraps the reference with FSDP too; mixed bf16/fp32 params fail flattening.
    ref_model = ref_model.to(dtype=dtype)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.config.use_cache = False
    ref_model.eval()

    return policy, ref_model, tok


# ====================================================================
# Main
# ====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pref_file",   required=True)
    p.add_argument("--val_pref_file", default=None)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--model_name",  default=DEFAULT_MODEL)
    p.add_argument("--sft_adapter", default=None,
                   help="Path to LoRA adapter from SFT stage; loaded as the "
                        "DPO policy and frozen as reference.")
    p.add_argument("--loss_type",   default="ipo",
                   choices=["sigmoid", "ipo", "hinge", "exo_pair", "kto_pair"])
    p.add_argument("--beta",        type=float, default=0.1)
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--max_steps",   type=int,   default=-1,
                   help="Optional hard cap on optimizer steps for pilot runs. "
                        "Leave -1 for normal epoch-based training.")
    p.add_argument("--batch_per_gpu", type=int, default=4)
    p.add_argument("--grad_accum",    type=int, default=4)
    p.add_argument("--lr",          type=float, default=5e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps",    type=int, default=500)
    p.add_argument("--eval_steps",    type=int, default=500)
    p.add_argument("--max_length",     type=int, default=4096)
    p.add_argument("--max_prompt_length", type=int, default=3072)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--lora_r",         type=int, default=64)
    p.add_argument("--lora_alpha",     type=int, default=128)
    p.add_argument("--lora_dropout",   type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing for DPO/IPO policy.")
    p.add_argument("--use_prm_weight", action="store_true",
                   help="Scale each preference pair's loss by its mean PRM score on the chosen trace.")
    p.add_argument("--prm_weight_fallback", default="sample",
                   choices=["sample", "error"],
                   help="When TRL does not expose a per-example loss hook, "
                        "'sample' uses deterministic PRM-weighted importance "
                        "sampling; 'error' fails loudly instead.")
    p.add_argument("--no_trl", action="store_true",
                   help="Force the manual fallback path (no TRL).")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="Path to a checkpoint dir, or 'auto' to resume from the "
                        "latest checkpoint inside --output_dir if one exists.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not _HAVE_HF:
        raise RuntimeError(
            f"transformers/peft not installed: {_HF_IMPORT_ERR}.  "
            f"Install on the cluster; this script is designed to be runnable "
            f"only where the full GPU stack is available."
        )

    policy, ref_model, tok = _build_model_and_ref(args)

    train_ds = PreferenceJsonlDataset(args.pref_file, tokenizer=tok)
    val_ds = (PreferenceJsonlDataset(args.val_pref_file, tokenizer=tok)
              if args.val_pref_file else None)
    print(f"[dpo] train preferences: {len(train_ds):,}")
    if val_ds:
        print(f"[dpo] val   preferences: {len(val_ds):,}")

    # Preference-type histogram for sanity
    hist: dict[str, int] = defaultdict(int)
    for rec in train_ds.records:
        hist[rec.get("mirror_type", "unknown")] += 1
    print(f"[dpo] preference-type histogram: {dict(hist)}")

    if _HAVE_TRL and not args.no_trl:
        trainer = _build_trl_trainer(policy, ref_model, tok, train_ds, val_ds, args)
        resume = args.resume_from_checkpoint
        if resume in ("", "none", "None", "false", "False"):
            resume = None
        if resume == "auto":
            ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"))
            resume = str(ckpts[-1]) if ckpts else None
            if resume:
                print(f"[dpo] auto-resume from {resume}")
            else:
                print("[dpo] auto-resume requested but no checkpoint found; starting fresh")
        trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(args.output_dir)
    else:
        print("[dpo] TRL not available (or --no_trl set); using manual path.")
        mt = _ManualDpoTrainer(policy, ref_model, tok, train_ds, val_ds, args)
        mt.train()

    print(f"[dpo] done.  adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
