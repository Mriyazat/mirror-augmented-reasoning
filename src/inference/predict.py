"""Generate canonical-format predictions from a Phase C SFT/DPO adapter.

Consumes a teacher-schema JSONL (same format as Phase C train/val: each
record carries `messages` with a system + user turn -- the assistant
turn is what the student produces), and writes a predictions JSONL
that `src.evaluation.run_full_eval` can score end-to-end.

Record schema (out)
-------------------
    {
      "pair_id":      str,
      "input_order":  "ab",     # extend with mirror pass separately
      "context_ids":  [str],    # copied from input record
      "trace": {
        "steps":        [ ... parsed from model JSON ... ],
        "final_answer": { family, subtype, direction_tag, polarity,
                          confidence, abstain, summary }
      },
      "final_prediction": {
        "family":        str,
        "subtype":       str,
        "direction_tag": str,
        "polarity":      str | null,
        "abstain":       bool,
        "confidence":    float | null,
        "label_dist":    {}   # populated later by classifier head (C5)
      },
      "raw_output":   str,    # the raw assistant string, for debugging
      "parse_ok":     bool,
      "parse_error":  str | null
    }

Usage
-----
    python -m src.inference.predict \\
        --adapter   ddi_checkpoints_v4/student/ddi_v4_sft_reasoning_safe \\
        --input     outputs/phase_c/teacher_clean.reasoning_safe.val.jsonl \\
        --output    outputs/phase_c/predictions_val.jsonl \\
        --max_new_tokens 768 --batch 4

    # Mirror pass (for MPS / CSA): flip pair order before generating.
    python -m src.inference.predict \\
        --adapter  ... \\
        --input    outputs/phase_c/teacher_clean.reasoning_safe.val.jsonl \\
        --output   outputs/phase_c/predictions_val.ba.jsonl \\
        --mirror
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

try:
    import torch
    _HAVE_TORCH = True
except Exception as _terr:
    _HAVE_TORCH = False
    _TORCH_ERR = _terr

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    _HAVE_HF = True
except Exception as _herr:
    _HAVE_HF = False
    _HF_ERR = _herr


# Lazy ContextBuilder. Loads ~6 parquets the first time it's used (~20s cold),
# but each pair_id is O(1) afterwards because we cache per-pair results.
# Required for MFS / HR / RIS to score above 0; without it those metrics
# silently report zero (the source of the v1 "MFS=0" reporting bug).
_CTX_BUILDER = None


def _ensure_context_builder():
    global _CTX_BUILDER
    if _CTX_BUILDER is None:
        from src.teacher.context_builder import ContextBuilder
        _CTX_BUILDER = ContextBuilder()
    return _CTX_BUILDER


_CTX_CACHE: dict[str, list[str]] = {}
_CTX_STATS = {"built": 0, "cached": 0, "failed": 0}


def _get_context_ids(pair_id: str) -> list[str]:
    """Resolve the retrieval bundle for `pair_id`. Empty list on failure
    (a missing pair_id, builder load error, etc.) so the caller never crashes
    the run mid-batch."""
    if not pair_id:
        return []
    if pair_id in _CTX_CACHE:
        _CTX_STATS["cached"] += 1
        return _CTX_CACHE[pair_id]
    try:
        cb = _ensure_context_builder()
        ids = sorted(cb.build(pair_id).context_ids())
        _CTX_CACHE[pair_id] = ids
        _CTX_STATS["built"] += 1
        return ids
    except Exception:
        _CTX_CACHE[pair_id] = []
        _CTX_STATS["failed"] += 1
        return []


# ---------------------------------------------------------- JSON extraction
_JSON_START_RE = re.compile(r"\{")


def _extract_json(text: str) -> tuple[dict | None, str | None]:
    """Find the first balanced JSON object in text.

    Strategy: scan for the first '{', then walk forward tracking brace depth
    while respecting string escapes. Accept the first object that parses
    cleanly with `json.loads`. Falls back to a more lenient cleanup pass
    (strip trailing commas) before giving up.
    """
    starts = [m.start() for m in _JSON_START_RE.finditer(text)]
    for s in starts:
        depth = 0
        in_str = False
        esc = False
        for i in range(s, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[s:i + 1]
                        try:
                            return json.loads(candidate), None
                        except json.JSONDecodeError:
                            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                            try:
                                return json.loads(cleaned), None
                            except json.JSONDecodeError as e:
                                return None, f"{e.msg} at pos {e.pos}"
    return None, "no JSON object found"


# ---------------------------------------------------------- trace post-process
def _normalize_trace(parsed: dict, input_record: dict) -> dict:
    """Pull parsed JSON into the the canonical trace + final_prediction shape expected
    by src.evaluation.run_full_eval. Missing/malformed fields get safe
    defaults so the record still reaches the metrics.
    """
    steps = parsed.get("steps") or []
    if not isinstance(steps, list):
        steps = []
    fa = parsed.get("final_answer") or {}
    if not isinstance(fa, dict):
        fa = {}
    abstain = bool(fa.get("abstain", False))

    final_pred = {
        "family":        fa.get("family"),
        "subtype":       fa.get("subtype"),
        "direction_tag": fa.get("direction_tag", "bidirectional"),
        "polarity":      fa.get("polarity"),
        "abstain":       abstain,
        "confidence":    fa.get("confidence"),
        "label_dist":    {},
    }
    return {
        "trace": {"steps": steps, "final_answer": fa},
        "final_prediction": final_pred,
    }


# The the teacher context template (src/teacher/prompt.py :: render_user)
# emits a QUERY PAIR block of exactly two lines:
#     "  A = {name}  ({db_id})  ATC={atc}  MW={mw}  t_half={hl}"
#     "  B = {name}  ({db_id})  ATC={atc}  MW={mw}  t_half={hl}"
# All trailing "ATC/MW/t_half" metadata is drug-specific and must swap with the
# line body. The capture group (3) picks up everything after the paren.
_QP_A_LINE_RE = re.compile(r"^  A = (.+?)  \((DB\w+)\)(.*)$", re.M)
_QP_B_LINE_RE = re.compile(r"^  B = (.+?)  \((DB\w+)\)(.*)$", re.M)

# Drug-specific mechanism-of-action blocks. Body starts on the next line
# (2-space indent) and ends at either a blank line or the next "[" header.
_MOA_BLOCK_RE = re.compile(
    r"^\[Drug ([AB]) mechanism of action\]\n(  .+?)(?=\n\n|\n\[)",
    re.M | re.S,
)

# Sentinels. Control chars <= \x08 are safe: they never occur in the
# natural-language content that DrugBank / UniProt contribute.
_S_QP_A   = "\x01S_QP_A\x01"
_S_QP_B   = "\x02S_QP_B\x02"
_S_MOA_A  = "\x03S_MOA_A\x03"
_S_MOA_B  = "\x04S_MOA_B\x04"
_S_AN     = "\x05S_AN\x05"
_S_BN     = "\x06S_BN\x06"
_S_PID    = "\x07S_PID\x07"


# Mirror-pass diagnostics (populated while generating, logged at the end).
_MIRROR_STATS = {
    "attempted":           0,
    "ok":                  0,
    "noop_no_pipe":        0,
    "no_query_pair_block": 0,
    "no_moa_block":        0,
    "name_sub_lines_a":    0,
    "name_sub_lines_b":    0,
}


def _mirror_user_content(content: str, pair_id: str) -> tuple[str, bool, dict]:
    """Structurally swap A and B inside a teacher-format user prompt.

    Four position-dependent regions are exchanged (everything else is
    symmetric and stays put):

      1. QUERY PAIR A / B lines: entire line bodies (name, id, ATC, MW, t_half)
      2. [Drug A mechanism of action] / [Drug B mechanism of action]: body text
      3. `  A=<a_name>(:| protein-bullet | ...)` prefix lines -> `  A=<b_name>...`
         `  B=<b_name>(:| protein-bullet | ...)` prefix lines -> `  B=<a_name>...`
         (covers Active PK flags, Per-drug pathways, Per-drug proteins)
      4. Literal pair_id "DB_a|DB_b" occurrences (belts & suspenders; the
         the template does not actually emit it inside the user prompt)

    Symmetric sections that are left untouched:
      - Header labels `[Drug A ...]` / `[Drug B ...]` (keep positional meaning)
      - Shared pathways, Shared proteins (unordered sets)
      - Pair similarity signatures (symmetric scalars)
      - Top mechanistic-neighbor pairs (unrelated pair_ids; must not be
        rewritten because their own direction encoding lives inside them)

    Returns (new_content, ok, stats). On parse failure returns the original
    content with ok=False so the caller can bail cleanly.
    """
    stats = {
        "no_query_pair_block": 0,
        "no_moa_block":        0,
        "name_sub_lines_a":    0,
        "name_sub_lines_b":    0,
    }
    ma = _QP_A_LINE_RE.search(content)
    mb = _QP_B_LINE_RE.search(content)
    if ma is None or mb is None:
        stats["no_query_pair_block"] = 1
        return content, False, stats

    a_name, a_id, a_meta = ma.group(1), ma.group(2), ma.group(3)
    b_name, b_id, b_meta = mb.group(1), mb.group(2), mb.group(3)

    # ---- (1) QUERY PAIR line body swap via sentinel ----
    out = _QP_A_LINE_RE.sub(f"  A = {_S_QP_A}", content, count=1)
    out = _QP_B_LINE_RE.sub(f"  B = {_S_QP_B}", out, count=1)

    # ---- (2) MoA block body swap via sentinel ----
    moa_bodies: dict[str, str] = {}
    for m in _MOA_BLOCK_RE.finditer(out):
        moa_bodies[m.group(1)] = m.group(2)

    if "A" in moa_bodies and "B" in moa_bodies:
        def _moa_sub(m: "re.Match") -> str:
            body = m.group(2)
            sent = _S_MOA_A if m.group(1) == "A" else _S_MOA_B
            return m.group(0).replace(body, sent, 1)
        out = _MOA_BLOCK_RE.sub(_moa_sub, out)
    else:
        stats["no_moa_block"] = 1
        moa_bodies = {}

    # ---- (3) Name swap in "  A=<a_name>..." / "  B=<b_name>..." lines ----
    # Order longest drug-name first in case one is a substring of the other
    # (e.g. "Perphenazine" vs "Perphenazine enanthate"); regex is anchored
    # to the "  A=" / "  B=" prefix, so it never bleeds into MoA narrative.
    prefix_subs = sorted(
        [("A", a_name, _S_AN), ("B", b_name, _S_BN)],
        key=lambda t: -len(t[1]),
    )
    for prefix, name, tok in prefix_subs:
        pattern = re.compile(
            rf"^(  {prefix}=){re.escape(name)}", re.M,
        )
        out, n = pattern.subn(f"\\g<1>{tok}", out)
        stats[f"name_sub_lines_{prefix.lower()}"] = n

    # ---- (4) literal pair_id swap (usually a no-op; guardrail only) ----
    if pair_id and "|" in pair_id:
        flipped = f"{b_id}|{a_id}"
        if pair_id in out and pair_id != flipped:
            out = out.replace(pair_id, _S_PID)
            out = out.replace(_S_PID, flipped)

    # ---- substitute sentinels with swapped values ----
    out = out.replace(_S_QP_A, f"{b_name}  ({b_id}){b_meta}")
    out = out.replace(_S_QP_B, f"{a_name}  ({a_id}){a_meta}")
    if moa_bodies:
        out = out.replace(_S_MOA_A, moa_bodies["B"])
        out = out.replace(_S_MOA_B, moa_bodies["A"])
    out = out.replace(_S_AN, b_name)
    out = out.replace(_S_BN, a_name)

    return out, True, stats


def _build_prompt_messages(rec: dict, mirror: bool) -> list[dict]:
    """Return the prompt (system + user) messages for `rec`, optionally
    mirrored (present the same pair as (B, A)). See `_mirror_user_content`
    for the per-region swap semantics."""
    msgs = [m for m in rec.get("messages", []) if m.get("role") != "assistant"]
    if not mirror:
        return msgs

    pid = rec.get("pair_id") or ""
    _MIRROR_STATS["attempted"] += 1
    if "|" not in pid:
        _MIRROR_STATS["noop_no_pipe"] += 1
        return msgs

    out = []
    for m in msgs:
        if m.get("role") == "user":
            new_content, ok, stats = _mirror_user_content(
                m.get("content") or "", pid,
            )
            if ok:
                _MIRROR_STATS["ok"] += 1
                _MIRROR_STATS["name_sub_lines_a"] += stats["name_sub_lines_a"]
                _MIRROR_STATS["name_sub_lines_b"] += stats["name_sub_lines_b"]
                if stats["no_moa_block"]:
                    _MIRROR_STATS["no_moa_block"] += 1
            else:
                _MIRROR_STATS["no_query_pair_block"] += 1
            out.append({"role": "user", "content": new_content})
        else:
            out.append(m)
    return out


# ---------------------------------------------------------- model I/O
def _load_model(base_model: str, adapter_dir: str, dtype_str: str,
                device_map: str = "auto"):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_str]
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # critical for decoder-only generation

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
    if adapter_dir:
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        model = base
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------- main generation
def _iter_jsonl(path: str | Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int,
                    do_sample: bool, temperature: float, top_p: float) -> list[str]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=4096)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)
    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    # Strip the prompt tokens to leave only the generated continuation
    input_len = enc["input_ids"].shape[1]
    gen = out[:, input_len:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None,
                   help="Path to PEFT adapter dir. Omit to run the base model.")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--input", required=True,
                   help="Input JSONL with teacher-style `messages`.")
    p.add_argument("--output", required=True,
                   help="Output JSONL in the canonical predictions format.")
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only predict the first N records.")
    p.add_argument("--do_sample", action="store_true",
                   help="Sample (temperature>0). Default is greedy.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--mirror", action="store_true",
                   help="Mirror pass: flip pair_id order in the user prompt, "
                        "and emit input_order='ba' in the output.")
    p.add_argument("--input_order", default=None,
                   help="Override input_order tag (auto: 'ba' if --mirror else 'ab').")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--resume", action="store_true",
                   help="If the output file already exists, skip any pair_id "
                        "that has already been written and append new "
                        "predictions instead of overwriting.")
    p.add_argument("--with_context", dest="with_context", action="store_true",
                   help="Resolve and emit context_ids per pair via "
                        "ContextBuilder. Required for MFS / HR / RIS to be "
                        "non-zero in eval. Default on.")
    p.add_argument("--no_context", dest="with_context", action="store_false",
                   help="Disable context_ids resolution (faster, but MFS / "
                        "HR / RIS will report 0).")
    p.set_defaults(with_context=True)
    args = p.parse_args()

    if not (_HAVE_TORCH and _HAVE_HF):
        raise SystemExit(
            "torch / transformers / peft required. "
            f"torch ok={_HAVE_TORCH}, hf ok={_HAVE_HF}"
        )

    input_order = args.input_order or ("ba" if args.mirror else "ab")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model(
        args.base_model, args.adapter, args.torch_dtype, args.device_map,
    )

    records = list(_iter_jsonl(args.input))
    if args.limit > 0:
        records = records[:args.limit]

    # ---- optional resume: skip pair_ids that already appear in out_path ----
    done_ids: set[str] = set()
    open_mode = "w"
    if args.resume and out_path.exists():
        try:
            with out_path.open() as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # Last line may be truncated if previous run was killed;
                        # drop it rather than crash.
                        continue
                    pid = rec.get("pair_id")
                    if pid:
                        done_ids.add(pid)
            open_mode = "a"
            records = [r for r in records if r.get("pair_id") not in done_ids]
            print(f"[predict] --resume: kept {len(done_ids)} prior records, "
                  f"{len(records)} still to predict.")
        except Exception as e:
            # Never block a run just because the previous output file is weird.
            print(f"[predict] --resume: failed to read {out_path} "
                  f"({type(e).__name__}: {e}); overwriting instead.")
            done_ids = set()
            open_mode = "w"

    print(f"[predict] {len(records)} input records, "
          f"batch={args.batch}, mirror={args.mirror}")

    n_parse_ok = 0
    n_parse_fail = 0
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
            gens = _generate_batch(
                model, tokenizer, prompts,
                args.max_new_tokens, args.do_sample,
                args.temperature, args.top_p,
            )
            for rec, raw in zip(chunk, gens):
                parsed, err = _extract_json(raw)
                if parsed is None:
                    parsed = {}
                    err = err or "no_json"
                    n_parse_fail += 1
                else:
                    n_parse_ok += 1
                shaped = _normalize_trace(parsed, rec)

                pair_id = rec.get("pair_id", "")
                ctx_ids = rec.get("context_ids") or []
                if args.with_context and not ctx_ids:
                    ctx_ids = _get_context_ids(pair_id)

                out_rec = {
                    "pair_id":      pair_id,
                    "input_order":  input_order,
                    "context_ids":  ctx_ids,
                    "trace":        shaped["trace"],
                    "final_prediction": shaped["final_prediction"],
                    "raw_output":   raw,
                    "parse_ok":     err is None,
                    "parse_error":  err,
                }
                fout.write(json.dumps(out_rec) + "\n")

            done = min(i + args.batch, len(records))
            if done % (args.batch * 20) == 0 or done == len(records):
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                print(f"[predict]   {done}/{len(records)}  "
                      f"parse_ok={n_parse_ok} fail={n_parse_fail}  "
                      f"({rate:.2f} rec/s)")

    print(f"[predict] done. parse_ok={n_parse_ok} "
          f"parse_fail={n_parse_fail} out={out_path}")
    if args.with_context:
        print(f"[predict] context_ids stats: {_CTX_STATS}")
        if _CTX_STATS["built"] + _CTX_STATS["cached"] == 0:
            print("[predict] WARNING: 0 context_ids resolved. Eval MFS/HR/RIS "
                  "will be 0. Check ContextBuilder data paths.")
        elif _CTX_STATS["failed"] > 0:
            print(f"[predict] WARNING: {_CTX_STATS['failed']} pair_ids failed "
                  "context resolution; their MFS/HR/RIS will be 0.")
    if args.mirror:
        print(f"[predict] mirror stats: {_MIRROR_STATS}")
        if _MIRROR_STATS["no_query_pair_block"] > 0:
            print("[predict] WARNING: some records could not be structurally "
                  "mirrored (missing 'A = ...' / 'B = ...' anchors). "
                  "Those records were sent with the original prompt; their "
                  "input_order='ba' label is therefore misleading. "
                  "Check the upstream teacher-context template.")


if __name__ == "__main__":
    main()
