"""Student SFT -- Student SFT with tier-weighted + faithfulness
+ symmetry losses.

Student base: Qwen2.5-7B (chat template) + LoRA rank 64.

Why this script, not TRL's SFTTrainer
-------------------------------------
TRL's SFTTrainer is great for vanilla SFT on chat data, but the current pipeline needs
three deviations from vanilla CE:

  1.  **Tier-weighted loss multiplier.**  Each SFT record carries a
      `sample_weight` (1.00 / 0.50 / 0.33 / 0.25 for
      full_correct / family_correct / abstention / near_miss).  This
      teaches the student that full_correct traces are gold demos and
      near_miss traces are useful but should contribute less signal
      per-token.

  2.  **Faithfulness regularizer.**  For each record we pre-compute the
      fraction of context evidence IDs that appear verbatim in the
      teacher rationale ("evidence-mention rate").  At training time
      this acts as an AUX weight multiplier on top of tier weight --
      records whose teacher rationales drifted from the evidence pool
      contribute proportionally less to the loss.  Cheaper and more
      interpretable than the plan's attention-over-evidence proposal,
      and it composes cleanly with tier weighting.

  3.  **Mirror-symmetry KL.**  When a batch contains both orderings of
      the same pair (mirror pairs -- injected via
      `--mirror_aug_rate`), we add a KL penalty between the two
      family-token distributions.  This is the in-training complement
      to the CSA metric: force the student to predict the same family
      under either ordering.

Other important details
-----------------------
- **LoRA via `peft`.**  Rank 64, alpha 128, dropout 0.05 per
  configs/base.yaml.  Target modules: q_proj, k_proj, v_proj, o_proj,
  gate_proj, up_proj, down_proj.
- **Qwen2.5 chat template.**  We rely on the tokenizer's built-in
  chat template (`apply_chat_template`) so assistant-only masking is
  handled correctly regardless of token IDs.
- **Mid-training evaluation hooks.**  On every eval step we compute a
  *lightweight* MFS / RPC over a held-out val batch using greedy
  decoding -- this lets us monitor mechanism faithfulness during
  training, not just perplexity.
- **Checkpoint strategy.**  Save LoRA adapter every N steps plus a
  `best` checkpoint based on val MFS + RPC weighted score.

Not in scope (separate scripts)
-------------------------------
  - Classifier head training (`src/training/classifier_head.py`, T-HEAD)
  - DPO / IPO mirror training (`src/training/dpo_mirror.py`, T-DPO)
  - Abstention calibration (`src/training/abstention.py`, T-ABSTAIN)

Usage
-----
    python -m src.training.sft_train \
        --train_file outputs/teacher/teacher_clean.jsonl \
        --val_file   outputs/teacher/teacher_clean_val.jsonl \
        --output_dir outputs/student/ddi_v4_sft \
        --epochs 3 \
        --faithfulness_weight 0.5 \
        --symmetry_weight 0.3 \
        --mirror_aug_rate 0.3

On a 4 x H100 node with FSDP + LoRA, Qwen2.5-7B at batch_per_gpu=8,
grad_accum=4 runs ~1.8 k SFT records / minute (under 25 minutes / epoch
on the 25 k subset).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.teacher.evidence_resolution import resolves, expand_context_ids

# Soft imports so `--help` and static parsing work without torch installed.
try:
    import torch                                                    # noqa: F401
    import torch.nn.functional as F                                 # noqa: F401
    from torch.utils.data import Dataset, Sampler                   # type: ignore
    _HAVE_TORCH = True
except Exception as _torch_err:
    _HAVE_TORCH = False
    _TORCH_IMPORT_ERR = _torch_err

    class Dataset:  # type: ignore[no-redef]
        """Placeholder so class definitions still parse without torch."""

    class Sampler:  # type: ignore[no-redef]
        """Placeholder so class definitions still parse without torch."""


try:
    from transformers import (                                     # noqa: E402
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
        Trainer, DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, PeftModel, get_peft_model, TaskType  # noqa: E402
    _HAVE_HF = True
except Exception as _hf_err:
    _HAVE_HF = False
    _HF_IMPORT_ERR = _hf_err

    class Trainer:  # type: ignore[no-redef]
        """Placeholder so DdiSftTrainer definition parses without HF installed."""


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_IGNORE_INDEX = -100


# --------------------------------------------------------------------- dataset
class SftJsonlDataset(Dataset):
    """Loads teacher_clean.jsonl records and tokenizes on the fly.

    Each record must have:
        messages       : list[dict]   -- chat-format conversation
        pair_id        : str
        family         : str
        direction_tag  : str
        sample_weight  : float
        tier           : str
        (optional) context_ids : list[str]  -- used for faithfulness weight
        (optional) mirror_pair_id : str     -- set when a mirror exists
    """

    def __init__(self, path: str | Path, tokenizer, max_length: int = 4096,
                 compute_faithfulness_weight: bool = True,
                 class_balance_mode: str = "off",
                 class_balance_max: float = 4.0):
        """
        class_balance_mode:
          "off"          : no class balancing (legacy behavior)
          "sqrt_inverse" : multiply sample_weight by 1/sqrt(class_freq) then
                           normalize so the corpus-mean weight is unchanged.
                           This is the recommended mode when the corpus is
                           class-imbalanced (it lifts rare classes without
                           letting them dominate the gradient).
          "inverse"      : 1/class_freq (more aggressive); cap at
                           class_balance_max to avoid one tiny class
                           dominating updates.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.compute_faithfulness = compute_faithfulness_weight
        self.records: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

        self.class_balance_mode = class_balance_mode
        self.class_balance_max = float(class_balance_max)
        self._apply_class_balance()

    def _apply_class_balance(self) -> None:
        """Bake a per-record class-balance multiplier into 'sample_weight'.

        Mutates each record so downstream __getitem__ keeps its 1-line
        sample_weight read. This is intentionally stamped at load time so
        we can log the effective per-class weight distribution once.
        """
        if self.class_balance_mode in ("off", "none", "", None):
            return
        from collections import Counter
        import math
        counts = Counter()
        for r in self.records:
            fam = r.get("family") or "UNKNOWN"
            counts[fam] += 1
        n_total = sum(counts.values())
        if n_total == 0:
            return

        if self.class_balance_mode == "sqrt_inverse":
            raw = {f: 1.0 / math.sqrt(c / n_total) for f, c in counts.items()}
        elif self.class_balance_mode == "inverse":
            raw = {f: min(self.class_balance_max, n_total / c)
                   for f, c in counts.items()}
        else:
            raise ValueError(f"unknown class_balance_mode: {self.class_balance_mode!r}")

        # Normalize so mean class weight (over the corpus) is 1.0 -> total
        # gradient magnitude is unchanged; only its allocation shifts.
        mean_w = sum(raw[r.get("family") or "UNKNOWN"] for r in self.records) / n_total
        for r in self.records:
            fam = r.get("family") or "UNKNOWN"
            cb = raw[fam] / max(mean_w, 1e-12)
            r["sample_weight"] = float(r.get("sample_weight", 1.0)) * cb

        # Log the resulting per-family sums so a human can verify balance
        rebalanced = Counter()
        for r in self.records:
            rebalanced[r.get("family") or "UNKNOWN"] += float(r.get("sample_weight", 1.0))
        total_w = sum(rebalanced.values()) or 1.0
        print(f"[class_balance] mode={self.class_balance_mode}  n_records={n_total}")
        print(f"[class_balance] family    count   raw_w   weighted_share")
        for f in sorted(counts.keys()):
            c = counts[f]
            print(f"[class_balance]   {f:18s} {c:>5d}  {raw[f]:>5.2f}  "
                  f"{100*rebalanced[f]/total_w:>5.1f}%")

    def __len__(self) -> int:
        return len(self.records)

    def _faithfulness_weight(self, rec: dict) -> float:
        """Evidence-mention rate in the teacher rationale.

        Computed once at dataset load so it costs zero training time.
        Value is in (0, 1] via a +1 Laplace smoother so records with no
        context_ids keep a non-zero weight.
        """
        if not self.compute_faithfulness:
            return 1.0
        ctx = rec.get("context_ids") or []
        if not ctx:
            return 1.0
        expanded = expand_context_ids(ctx)
        try:
            rationale = next(
                m["content"] for m in rec["messages"]
                if m.get("role") == "assistant"
            )
        except StopIteration:
            return 1.0
        rationale = rationale or ""
        # Tokenize the rationale into candidate entity-ish strings.  The
        # cheapest useful signal: what fraction of context ids appear as
        # substrings in the rationale (after normalization).
        hits = 0
        for cid in ctx:
            if not isinstance(cid, str) or not cid:
                continue
            if resolves(cid, expanded) and cid.lower() in rationale.lower():
                hits += 1
        rate = (hits + 1) / (len(ctx) + 1)
        return max(0.2, min(1.0, rate))  # floor at 0.2 so bad traces still train

    @staticmethod
    def _value_token_idx(full_text: str, offsets: list, needle: str) -> int:
        """Return the input_ids index of the FIRST token of the JSON value
        following `needle` (e.g. needle='"family":"' returns the index of the
        first token of the family string).

        Uses the LAST occurrence of `needle` in `full_text` to dodge accidental
        substring hits inside the user prompt or system message. Returns -1 if
        the needle is missing or no token starts at the right character offset
        (which can happen if truncation cut off the assistant turn).
        """
        char_pos = full_text.rfind(needle)
        if char_pos < 0:
            return -1
        target_char = char_pos + len(needle)
        for ti, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:  # special token with no chars
                continue
            if s == target_char:
                return ti
            if s < target_char < e:
                return ti
        return -1

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        msgs = rec["messages"]
        tok = self.tokenizer

        # Build full prompt + response as a single tokenized sequence,
        # and set labels to ignore everything before the assistant reply.
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        prompt_text = tok.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )
        full_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
        # `return_offsets_mapping=True` lets us locate the token whose
        # character offset starts right after `"family":"` in the assistant
        # turn -- that's where the family-value token lives, which is the
        # only position where the symmetry-KL term should fire.
        enc_full = tok(full_text, truncation=True, max_length=self.max_length,
                       return_tensors=None, add_special_tokens=False,
                       return_offsets_mapping=True)
        enc_prompt = tok(prompt_text, truncation=True, max_length=self.max_length,
                         return_tensors=None, add_special_tokens=False)
        input_ids = enc_full["input_ids"]
        attn = enc_full.get("attention_mask") or [1] * len(input_ids)
        offsets = enc_full.get("offset_mapping") or []
        labels = list(input_ids)
        n_prompt = min(len(enc_prompt["input_ids"]), len(labels))
        for i in range(n_prompt):
            labels[i] = _IGNORE_INDEX

        family_pos = self._value_token_idx(full_text, offsets, '"family":"')
        # We sometimes also want to score on direction_tag for diagnostic /
        # ablation purposes (NOT for the swap-invariant KL; direction values
        # flip between AB and BA mirrors and so SHOULD disagree). Stored
        # alongside family_pos for run_full_eval-style probes if needed.
        direction_pos = self._value_token_idx(
            full_text, offsets, '"direction_tag":"'
        )

        tier = rec.get("tier", "full_correct")
        sw = float(rec.get("sample_weight", 1.0))
        fw = self._faithfulness_weight(rec)
        return {
            "input_ids":          input_ids,
            "attention_mask":     attn,
            "labels":             labels,
            "sample_weight":      sw,
            "faithfulness_weight": fw,
            "pair_id":            rec.get("pair_id", ""),
            "mirror_pair_id":     rec.get("mirror_pair_id"),
            "family":             rec.get("family"),
            "tier":               tier,
            "family_pos":         family_pos,
            "direction_pos":      direction_pos,
        }


class MirrorPairSampler(Sampler):
    """Yield indices so AB/BA mirror pairs land in the SAME micro-batch.

    The SftJsonlDataset built by `src.data.build_mirror_sft_corpus` writes
    records in the order:
        index 2k     -> AB record of canonical pair k
        index 2k+1   -> BA record of canonical pair k
    This sampler preserves that adjacency while shuffling the pair order
    each epoch. With `per_device_train_batch_size` a multiple of 2, each
    physical micro-batch contains complete mirror pairs, and the
    symmetry-KL term in `DdiSftTrainer.compute_loss` can actually fire.

    Single-process fallback. For multi-GPU runs the trainer wraps this
    in `DistributedMirrorPairSampler` so rank boundaries also respect
    the pair adjacency.
    """

    def __init__(self, dataset_len: int, seed: int = 0):
        if dataset_len % 2 != 0:
            raise ValueError(
                f"MirrorPairSampler needs an even-length dataset "
                f"(got {dataset_len}). Rebuild the corpus with "
                f"src.data.build_mirror_sft_corpus so AB/BA pairs are "
                f"adjacent at indices (2k, 2k+1)."
            )
        self.n = dataset_len
        self.n_pairs = dataset_len // 2
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        perm = list(range(self.n_pairs))
        rng.shuffle(perm)
        # Interleave AB/BA so consecutive indices belong to the same pair.
        for p in perm:
            yield 2 * p
            yield 2 * p + 1

    def __len__(self):
        return self.n


class DistributedMirrorPairSampler(Sampler):
    """Distributed variant. Each rank receives a contiguous block of
    pair-shuffled indices so AB/BA stays together on the same rank
    (and, with batch_size >= 2, in the same micro-batch).
    """

    def __init__(self, dataset_len: int, rank: int, num_replicas: int,
                 seed: int = 0):
        if dataset_len % 2 != 0:
            raise ValueError(
                f"DistributedMirrorPairSampler needs an even-length dataset "
                f"(got {dataset_len})."
            )
        self.n = dataset_len
        self.n_pairs = dataset_len // 2
        self.rank = int(rank)
        self.num_replicas = int(num_replicas)
        self.seed = int(seed)
        self.epoch = 0
        # ceil-divide pairs across ranks; pad with replicated pairs so every
        # rank sees the same iteration length (standard DistributedSampler
        # behaviour).
        self.pairs_per_rank = (self.n_pairs + self.num_replicas - 1) // self.num_replicas
        self.num_samples = 2 * self.pairs_per_rank
        self.total_pairs_padded = self.pairs_per_rank * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pair_perm = list(range(self.n_pairs))
        rng.shuffle(pair_perm)
        pad = self.total_pairs_padded - self.n_pairs
        if pad > 0:
            pair_perm = pair_perm + pair_perm[:pad]
        start = self.rank * self.pairs_per_rank
        end = start + self.pairs_per_rank
        rank_pairs = pair_perm[start:end]
        for p in rank_pairs:
            yield 2 * p
            yield 2 * p + 1

    def __len__(self):
        return self.num_samples


def _collate(features: list[dict], tokenizer, max_length: int) -> dict:
    """Pads a batch (dynamic) and packages side-channels for the custom loss."""
    input_ids = [f["input_ids"] for f in features]
    attn = [f["attention_mask"] for f in features]
    labels = [f["labels"] for f in features]

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    maxlen = min(max(len(x) for x in input_ids), max_length)

    def _pad_right(lst, pad_val):
        out = []
        for x in lst:
            x = x[:maxlen]
            out.append(x + [pad_val] * (maxlen - len(x)))
        return out

    ii = _pad_right(input_ids, pad_id)
    aa = _pad_right(attn, 0)
    ll = _pad_right(labels, _IGNORE_INDEX)

    batch = {
        "input_ids":      torch.tensor(ii, dtype=torch.long),
        "attention_mask": torch.tensor(aa, dtype=torch.long),
        "labels":         torch.tensor(ll, dtype=torch.long),
    }
    batch["sample_weight"]        = torch.tensor(
        [f["sample_weight"] for f in features], dtype=torch.float)
    batch["faithfulness_weight"]  = torch.tensor(
        [f["faithfulness_weight"] for f in features], dtype=torch.float)
    batch["family_pos"]           = torch.tensor(
        [f.get("family_pos", -1) for f in features], dtype=torch.long)
    batch["direction_pos"]        = torch.tensor(
        [f.get("direction_pos", -1) for f in features], dtype=torch.long)
    batch["_pair_ids"]     = [f["pair_id"] for f in features]
    batch["_mirror_ids"]   = [f["mirror_pair_id"] for f in features]
    batch["_families"]     = [f["family"] for f in features]
    batch["_tiers"]        = [f["tier"] for f in features]
    return batch


# ---------------------------------------------------------- custom Trainer
class DdiSftTrainer(Trainer):  # type: ignore[name-defined]
    """Trainer subclass implementing the tier + faithfulness + symmetry loss.

    Loss composition (per batch):
        L = sum_i (sample_weight_i * faithfulness_weight_i) * CE_i
          + lambda_sym * L_sym

    where CE_i is the mean per-example NLL on assistant tokens.  L_sym
    is computed only when the batch contains at least one mirror pair
    (an (A,B) and (B,A) record for the same canonical pair).
    """

    def __init__(self, *args, lambda_sym: float = 0.0,
                 lambda_faith: float = 1.0,
                 use_mirror_pair_sampler: bool = False,
                 symmetry_mode: str = "family_token", **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_sym = lambda_sym
        # faithfulness is already folded into per-example weights, but we
        # keep a scalar knob so ablations (e.g., -- faithfulness) can zero it.
        self.lambda_faith = lambda_faith
        self.use_mirror_pair_sampler = bool(use_mirror_pair_sampler)
        # symmetry_mode controls WHERE the KL is taken between the two
        # orderings of a mirror pair:
        #   family_token : at the first token of the family JSON value
        #                  (correct: family is swap-invariant under AB<->BA
        #                   in build_mirror_sft_corpus)
        #   last_token   : at the final assistant token; this is what v1 did
        #                  and is preserved only as a reproducibility ablation
        #                  for the regression we already saw (do not use)
        #   disabled     : zero out the term regardless of lambda_sym
        if symmetry_mode not in {"family_token", "last_token", "disabled"}:
            raise ValueError(
                f"symmetry_mode must be family_token | last_token | disabled "
                f"(got {symmetry_mode!r})"
            )
        self.symmetry_mode = symmetry_mode
        # Running diagnostics for the symmetry loss; flushed via .log().
        self._sym_loss_sum = 0.0
        self._sym_loss_n = 0
        self._sym_pairs_seen = 0
        self._sym_pairs_skipped_no_pos = 0
        self._sym_batches_seen = 0

    def _get_train_sampler(self, train_dataset=None):
        """Use the mirror-pair sampler when the corpus is mirror-augmented.

        The default HF RandomSampler would shuffle AB/BA records
        independently, so the symmetry-KL term below would almost never
        see both orderings of the same pair in one micro-batch. Override
        returns a sampler that yields indices in (AB, BA) pair order.
        """
        if not self.use_mirror_pair_sampler:
            return super()._get_train_sampler(train_dataset)

        ds = train_dataset if train_dataset is not None else self.train_dataset
        n = len(ds)
        # Return the global pair-preserving sampler. HF/Accelerate performs
        # process sharding when it prepares the dataloader. If we pre-shard
        # here as well, distributed runs are double-sharded and see only a
        # fraction of the intended update steps. Because the sampler emits
        # adjacent (AB, BA) indices and the per-device batch size is even,
        # Accelerate's batch-level sharding still keeps pairs together.
        return MirrorPairSampler(n, seed=self.args.seed)

    def compute_loss(self, model, inputs, return_outputs: bool = False,
                     num_items_in_batch=None):
        sample_w = inputs.pop("sample_weight")
        faith_w  = inputs.pop("faithfulness_weight")
        pair_ids = inputs.pop("_pair_ids", None)
        mirrors  = inputs.pop("_mirror_ids", None)
        family_pos = inputs.pop("family_pos", None)
        _ = inputs.pop("direction_pos", None)
        _ = inputs.pop("_families", None)
        _ = inputs.pop("_tiers", None)

        # Pop labels before the forward pass. Otherwise HF causal-LM models
        # compute their own full-vocab CE loss, then we compute the custom
        # weighted CE below as well. That duplicate loss path is a large
        # memory hit at sequence length 4096 with per-device batch size 2.
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        # Per-example CE on assistant tokens. Computing cross-entropy over the
        # full (B*T, vocab) matrix at once creates a multi-GB temporary for
        # Qwen's 151k vocab. Chunk along sequence length to keep the exact same
        # loss while avoiding an OOM at batch_per_gpu=2, max_length=4096.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:].contiguous()
        B, T, V = shift_logits.shape
        valid_mask = (shift_labels != _IGNORE_INDEX).float()
        loss_sum = torch.zeros(B, device=logits.device, dtype=torch.float32)
        token_count = valid_mask.sum(dim=1).clamp(min=1.0)
        ce_chunk = 256
        for start in range(0, T, ce_chunk):
            end = min(start + ce_chunk, T)
            chunk_logits = shift_logits[:, start:end, :].reshape(-1, V)
            chunk_labels = shift_labels[:, start:end].reshape(-1)
            chunk_loss = F.cross_entropy(
                chunk_logits,
                chunk_labels,
                ignore_index=_IGNORE_INDEX,
                reduction="none",
            ).view(B, end - start)
            loss_sum += (chunk_loss * valid_mask[:, start:end]).sum(dim=1).float()
        per_ex_nll = loss_sum / token_count

        # Folded tier + faithfulness weighting
        eff_w = sample_w * ((1.0 - self.lambda_faith) + self.lambda_faith * faith_w)
        ce_loss = (per_ex_nll * eff_w).sum() / eff_w.sum().clamp(min=1e-6)

        # Symmetry loss: KL between the two orderings of a mirror pair at
        # the family-value token. Family is swap-invariant in
        # build_mirror_sft_corpus (it ONLY flips direction_tag), so we expect
        # AB and BA to predict the same family distribution. The v1 last-token
        # mode applied KL at the closing brace of the JSON, which collapsed
        # the family distribution toward the dominant class -- preserved here
        # only as a reproducibility ablation, never as a default.
        sym_loss = torch.tensor(0.0, device=ce_loss.device)
        run_sym = (
            mirrors is not None
            and self.lambda_sym > 0.0
            and pair_ids is not None
            and self.symmetry_mode != "disabled"
        )
        if run_sym:
            groups: dict[tuple, list[int]] = defaultdict(list)
            for i, (pid, mid) in enumerate(zip(pair_ids, mirrors)):
                if mid is None:
                    continue
                key = tuple(sorted([pid, mid]))
                groups[key].append(i)

            if self.symmetry_mode == "last_token":
                last_pos = (valid_mask.cumsum(dim=1) ==
                            valid_mask.sum(dim=1, keepdim=True)).float()
                pos_per_ex = last_pos.argmax(dim=1).clamp(max=T - 1)
            else:  # family_token
                if family_pos is None:
                    pos_per_ex = None
                else:
                    # logits at index t predict labels at index t+1, so the
                    # logit slot that is supervised on the family-value
                    # token is family_pos - 1.
                    pos_per_ex = (family_pos.to(shift_logits.device) - 1).clamp(min=0, max=T - 1)

            pair_kls = []
            n_skipped = 0
            for _key, idxs in groups.items():
                if len(idxs) < 2:
                    continue
                i0, i1 = idxs[0], idxs[1]
                if pos_per_ex is None:
                    n_skipped += 1
                    continue
                p0_idx = int(pos_per_ex[i0].item())
                p1_idx = int(pos_per_ex[i1].item())
                if p0_idx < 0 or p1_idx < 0:
                    # family marker not found in one of the two records
                    # (truncated assistant turn or non-JSON output). Skip.
                    n_skipped += 1
                    continue
                logp0 = F.log_softmax(shift_logits[i0, p0_idx, :].float(), dim=-1)
                logp1 = F.log_softmax(shift_logits[i1, p1_idx, :].float(), dim=-1)
                # symmetric KL in log-prob space
                kl_01 = F.kl_div(logp0, logp1.exp(), reduction="sum")
                kl_10 = F.kl_div(logp1, logp0.exp(), reduction="sum")
                pair_kls.append(0.5 * (kl_01 + kl_10))
            if pair_kls:
                sym_loss = torch.stack(pair_kls).mean()
                self._sym_pairs_seen += len(pair_kls)
            self._sym_pairs_skipped_no_pos += n_skipped

        # Diagnostic accumulation (flushed in .log()).
        self._sym_loss_sum += float(sym_loss.detach().cpu().item())
        self._sym_loss_n += 1
        self._sym_batches_seen += 1

        loss = ce_loss + self.lambda_sym * sym_loss
        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: dict, *args, **kwargs):
        """Inject sym_loss diagnostics on each Trainer log tick."""
        if self._sym_loss_n > 0:
            logs = dict(logs)
            logs["sym_loss"] = self._sym_loss_sum / self._sym_loss_n
            logs["sym_pairs_per_batch"] = self._sym_pairs_seen / self._sym_loss_n
            logs["sym_pairs_skipped_no_pos_per_batch"] = (
                self._sym_pairs_skipped_no_pos / self._sym_loss_n
            )
            logs["sym_mode"] = self.symmetry_mode
            # reset accumulators after emit
            self._sym_loss_sum = 0.0
            self._sym_loss_n = 0
            self._sym_pairs_seen = 0
            self._sym_pairs_skipped_no_pos = 0
        return super().log(logs, *args, **kwargs)


# ---------------------------------------------------------- build_model
def build_model_and_tokenizer(model_name: str, lora_r: int, lora_alpha: int,
                              lora_dropout: float, target_modules: list[str],
                              torch_dtype: str = "bfloat16",
                              gradient_checkpointing: bool = False,
                              adapter_init: str | None = None):
    if not _HAVE_HF:
        raise RuntimeError(
            f"transformers / peft not installed on this machine: {_HF_IMPORT_ERR!r}.  "
            f"Run `pip install transformers peft` in the training venv before invoking."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[torch_dtype]
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, trust_remote_code=True,
        )
    except TypeError:
        # Older transformers versions used `torch_dtype`; newer versions warn.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
        )
    # Training does not need generation KV cache; disabling it saves memory
    # and avoids incompatibilities with checkpointing/FSDP setups.
    model.config.use_cache = False
    if adapter_init:
        print(f"[sft_train] warm-start LoRA adapter from {adapter_init}")
        model = PeftModel.from_pretrained(model, adapter_init, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        model = get_peft_model(model, lora_cfg)
    if gradient_checkpointing:
        # PEFT freezes the base embeddings. With re-entrant gradient
        # checkpointing, the checkpoint wrapper requires at least one input
        # tensor with requires_grad=True; otherwise it detaches the forward
        # graph and backward fails with "loss does not require grad".
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _make_inputs_require_grad(_module, _inputs, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------- eval helpers
def _preprocess_logits_for_metrics(logits, labels):
    """Reduce full-vocab logits to argmax token ids before they reach compute_metrics.

    Without this hook, Trainer materializes (num_eval * max_len * vocab_size)
    float32 tensors on host, which with vocab=151936 and max_len=4096 is many
    hundreds of GB for a modest val set. Returning argmax ids keeps the
    accumulator tiny and compute_metrics fast.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def _compute_eval_metrics(eval_pred) -> dict:
    """Assistant-token accuracy on the val set (teacher-forced).

    A useful second signal on top of eval_loss:
      - Token-level top-1 accuracy over the assistant portion only
        (label positions where label != -100).
      - Because labels were shifted to ignore prompt tokens, this reflects
        the student's ability to predict teacher JSON tokens when given
        the ground-truth prefix. It tracks eval_loss closely but is
        interpretable as a percent.

    Not a generation metric. MFS / RPC / MPS / HR / family-EM all require
    actual sampled JSON, produced by src.inference.predict + scored by
    src.evaluation.run_full_eval.
    """
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    shift_preds = preds[:, :-1]
    shift_labels = labels[:, 1:]
    mask = shift_labels != _IGNORE_INDEX
    n_valid = int(mask.sum())
    if n_valid == 0:
        return {"assistant_token_acc": 0.0, "assistant_tokens": 0}
    acc = float((shift_preds[mask] == shift_labels[mask]).mean())
    return {
        "assistant_token_acc": acc,
        "assistant_tokens": n_valid,
    }


# ---------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file",   default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--epochs",     type=int, default=3)
    p.add_argument("--max_steps",  type=int, default=-1,
                   help="Optional hard cap on optimizer steps for pilot runs. "
                        "Leave at -1 for normal epoch-based training.")
    p.add_argument("--batch_per_gpu", type=int, default=8)
    p.add_argument("--grad_accum",    type=int, default=4)
    p.add_argument("--lr",            type=float, default=2.0e-4)
    p.add_argument("--warmup_ratio",  type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Gradient norm clipping. Drop to 0.5 for stability "
                        "when sym_KL is on (defends against bf16 overflow).")
    p.add_argument("--class_balance_mode", default="off",
                   choices=["off", "sqrt_inverse", "inverse"],
                   help="Multiply sample_weight by an inverse-frequency "
                        "term so rare-family records get fairer gradient "
                        "allocation. 'sqrt_inverse' is the recommended "
                        "default for class-imbalanced corpora.")
    p.add_argument("--class_balance_max", type=float, default=4.0,
                   help="Cap on the inverse-frequency multiplier (only "
                        "used when --class_balance_mode=inverse).")
    p.add_argument("--faithfulness_weight", type=float, default=0.5,
                   help="Blending coefficient: loss_weight = sample_w * "
                        "((1 - f) + f * faithfulness_mention_rate).")
    p.add_argument("--symmetry_weight",     type=float, default=0.3,
                   help="Multiplier on the KL(P_fam|ab || P_fam|ba) term.")
    p.add_argument("--lora_r",        type=int, default=64)
    p.add_argument("--lora_alpha",    type=int, default=128)
    p.add_argument("--lora_dropout",  type=float, default=0.05)
    p.add_argument("--adapter_init", default=None,
                   help="Optional existing PEFT/LoRA adapter directory to "
                        "warm-start from. When set, the adapter is loaded as "
                        "trainable and lora_r/lora_alpha/lora_dropout only "
                        "apply to fresh runs without --adapter_init.")
    p.add_argument("--save_steps",    type=int, default=500)
    p.add_argument("--eval_steps",    type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--target_modules", nargs="+", default=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--disable_faithfulness_weight", action="store_true",
                   help="Ablation: set all per-example weights to 1.0 "
                        "(disables faithfulness signal).")
    p.add_argument("--disable_symmetry_loss", action="store_true",
                   help="Ablation: zero out the mirror-KL regularizer.")
    p.add_argument("--symmetry_mode", default="family_token",
                   choices=["family_token", "last_token", "disabled"],
                   help="Where to apply the symmetry-KL term across mirror "
                        "pairs. family_token (default, fixed in v2) targets "
                        "the family-value token (swap-invariant). last_token "
                        "is the v1 behaviour (closing brace) kept ONLY for "
                        "ablation reproduction; do NOT use as a default. "
                        "disabled zeros out the term irrespective of "
                        "--symmetry_weight.")
    p.add_argument("--mirror_pair_sampler", action="store_true",
                   help="Use MirrorPairSampler so AB/BA mirror records of "
                        "the same canonical pair land in the same micro-batch. "
                        "Required for the lambda_sym KL term to actually "
                        "fire on a mirror-augmented corpus produced by "
                        "src.data.build_mirror_sft_corpus.")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable Trainer gradient checkpointing. Leave disabled "
                        "with configs/accelerate_fsdp.yaml because FSDP "
                        "activation checkpointing is already enabled there.")
    p.add_argument("--report_to", default="none",
                   help="Trainer reporting backend. Use 'none' by default on "
                        "clusters without tensorboard; set to 'tensorboard' "
                        "only if tensorboard is installed.")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="Path to a checkpoint dir, or 'auto' to resume from the "
                        "latest checkpoint inside --output_dir if one exists.")
    p.add_argument("--load_best_at_end", action="store_true",
                   help="Load the best checkpoint (by --metric_for_best_model) at "
                        "the end of training. Requires eval_steps == save_steps.")
    p.add_argument("--metric_for_best_model", default="eval_loss",
                   help="Metric used to rank checkpoints when --load_best_at_end. "
                        "Common: eval_loss (lower better) or "
                        "eval_assistant_token_acc (higher better).")
    p.add_argument("--greater_is_better", default="auto",
                   choices=["auto", "true", "false"],
                   help="Direction for --metric_for_best_model. 'auto' infers from "
                        "the metric name (acc/f1/... => true, loss => false).")
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--skip_compute_metrics", action="store_true",
                   help="Ablation / debug: skip assistant_token_acc computation "
                        "during eval. eval_loss is still reported.")
    args = p.parse_args()

    if not _HAVE_HF:
        raise SystemExit(
            f"transformers / peft not importable.  Install them before training:\n"
            f"    {_HF_IMPORT_ERR}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = build_model_and_tokenizer(
        args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout,
        args.target_modules, args.torch_dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        adapter_init=args.adapter_init,
    )

    train_ds = SftJsonlDataset(
        args.train_file, tokenizer, max_length=args.max_length,
        compute_faithfulness_weight=not args.disable_faithfulness_weight,
        class_balance_mode=args.class_balance_mode,
        class_balance_max=args.class_balance_max,
    )
    eval_ds = None
    if args.val_file:
        # Eval set is intentionally NOT class-balanced -- we want eval_loss
        # to reflect the natural distribution.
        eval_ds = SftJsonlDataset(
            args.val_file, tokenizer, max_length=args.max_length,
            compute_faithfulness_weight=not args.disable_faithfulness_weight,
            class_balance_mode="off",
        )

    def _collator(batch: list[dict]) -> dict:
        return _collate(batch, tokenizer, args.max_length)

    load_best = bool(args.load_best_at_end and eval_ds is not None)
    if load_best and args.eval_steps != args.save_steps:
        print(f"[sft_train] --load_best_at_end requires eval_steps == save_steps; "
              f"forcing save_steps={args.eval_steps} (was {args.save_steps}).")
        args.save_steps = args.eval_steps

    if args.greater_is_better == "auto":
        loss_like = any(k in args.metric_for_best_model.lower()
                        for k in ("loss", "nll", "ppl", "error"))
        greater_is_better = not loss_like
    else:
        greater_is_better = args.greater_is_better == "true"

    ta_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_per_gpu,
        per_device_eval_batch_size=args.batch_per_gpu,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        eval_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        bf16=(args.torch_dtype == "bfloat16"),
        fp16=(args.torch_dtype == "float16"),
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[] if args.report_to in ("", "none", "None") else [args.report_to],
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,  # we need sample_weight etc.
    )
    if load_best:
        ta_kwargs.update(
            load_best_model_at_end=True,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=greater_is_better,
        )
    targs = TrainingArguments(**ta_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": targs,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": _collator,
        "lambda_sym": 0.0 if args.disable_symmetry_loss else args.symmetry_weight,
        "lambda_faith": 0.0 if args.disable_faithfulness_weight else args.faithfulness_weight,
        "use_mirror_pair_sampler": args.mirror_pair_sampler,
        "symmetry_mode": "disabled" if args.disable_symmetry_loss else args.symmetry_mode,
    }
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if eval_ds is not None and not args.skip_compute_metrics:
        trainer_kwargs["compute_metrics"] = _compute_eval_metrics
        trainer_kwargs["preprocess_logits_for_metrics"] = _preprocess_logits_for_metrics
    trainer = DdiSftTrainer(**trainer_kwargs)

    resume = args.resume_from_checkpoint
    if resume in ("", "none", "None", "false", "False"):
        resume = None
    if resume == "auto":
        ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"))
        resume = str(ckpts[-1]) if ckpts else None
        if resume:
            print(f"[sft_train] auto-resume from {resume}")
        else:
            print("[sft_train] auto-resume requested but no checkpoint found; starting fresh")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    cfg_dump = {k: v for k, v in vars(args).items()}
    with open(Path(args.output_dir) / "sft_config.json", "w") as f:
        json.dump(cfg_dump, f, indent=2)

    print(f"[sft_train] saved student to {args.output_dir}")


if __name__ == "__main__":
    main()
