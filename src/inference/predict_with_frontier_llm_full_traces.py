"""Frontier-LLM zero-shot DDI prediction WITH FULL TRACE PRESERVATION.

Identical to `predict_with_frontier_llm.py` but:
  * ALWAYS saves the raw model output (not just on parse failure).
  * Saves the full parsed trace structure (steps + final_answer).

This is required for the trace-quality LLM-judge evaluation, where we need
the per-step reasoning, not just the final family/subtype prediction.

Usage:

    source .env.trace_eval

    python -m src.inference.predict_with_frontier_llm_full_traces \\
        --prompts outputs/eval_prompts/random_full_test.prompts.jsonl \\
        --manifest outputs/eval_prompts/random_full_test_5000_stratified.manifest.jsonl \\
        --pair_ids_file outputs/diag2/trace_quality/sampled_pair_ids.json \\
        --output  outputs/diag2/trace_quality/pred_claude_traces_200.jsonl \\
        --provider anthropic --model claude-sonnet-4-5 \\
        --temperature 0.0 --max_tokens 2048 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

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
    by_fam = defaultdict(list)
    for pid in pair_ids:
        if pid in truth:
            by_fam[truth[pid]].append(pid)
    rng = random.Random(seed)
    for v in by_fam.values():
        rng.shuffle(v)
    per_fam = max(1, n // len(by_fam))
    out = []
    for fam, ids in by_fam.items():
        out.extend(ids[:per_fam])
    rng.shuffle(out)
    return out[:n]


def call_anthropic(messages, model, max_tokens, temperature):
    import anthropic
    client = anthropic.Anthropic()
    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs = [m for m in messages if m["role"] != "system"]
    backoff = 1.0
    for attempt in range(8):
        try:
            resp = client.messages.create(
                model=model,
                system=sys_msg,
                messages=user_msgs,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text"), {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        except anthropic.RateLimitError as e:
            wait = None
            try:
                hdrs = getattr(getattr(e, "response", None), "headers", {}) or {}
                ra = hdrs.get("retry-after") or hdrs.get("Retry-After")
                if ra:
                    wait = float(ra) + 0.5
            except Exception:
                pass
            time.sleep(wait if wait else min(backoff, 60.0))
            backoff *= 2.0
        except anthropic.APIStatusError as e:
            if getattr(e, "status_code", None) in (429, 529):
                time.sleep(min(backoff, 60.0))
                backoff *= 2.0
                continue
            raise
        except anthropic.APIConnectionError:
            time.sleep(min(backoff, 30.0))
            backoff *= 2.0
    raise RuntimeError("[call_anthropic] exceeded retries on rate limit")


def call_google(messages, model, max_tokens, temperature, thinking_budget=None):
    """Google Gemini adapter using the new `google-genai` SDK.

    `thinking_budget`: if None (default), uses model default (Gemini 2.5
    models have implicit thinking enabled). If 0, disables thinking entirely
    (recommended for short-output JSON tasks like the LLM-as-judge calls,
    because the thinking tokens consume the output budget and truncate the
    visible response). If a positive int, uses that many thinking tokens.
    """
    from google import genai as new_genai
    from google.genai import types as new_types

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY)")
    client = new_genai.Client(api_key=api_key)
    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msg = "\n\n".join(m["content"] for m in messages if m["role"] != "system")

    config_kwargs = {
        "system_instruction": sys_msg if sys_msg else None,
        "temperature": float(temperature),
        "max_output_tokens": int(max_tokens),
    }
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = new_types.ThinkingConfig(thinking_budget=int(thinking_budget))

    backoff = 1.0
    for attempt in range(8):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_msg,
                config=new_types.GenerateContentConfig(**config_kwargs),
            )
            txt = resp.text or ""
            um = getattr(resp, "usage_metadata", None)
            usage = {
                "input_tokens": getattr(um, "prompt_token_count", 0) if um else 0,
                "output_tokens": getattr(um, "candidates_token_count", 0) if um else 0,
                "thinking_tokens": getattr(um, "thoughts_token_count", 0) if um else 0,
            }
            return txt, usage
        except Exception as e:
            es = str(e)
            wait = None
            m = re.search(r"retry.{0,40}?(\d+(?:\.\d+)?)\s*s", es, re.I)
            if m:
                wait = float(m.group(1)) + 0.5
            if ("429" in es or "rate" in es.lower() or "quota" in es.lower()
                    or "RESOURCE_EXHAUSTED" in es):
                time.sleep(wait if wait else min(backoff, 30.0))
                backoff *= 2.0
                continue
            raise
    raise RuntimeError("[call_google] exceeded retries on rate limit")


def call_openai(messages, model, max_tokens, temperature):
    from openai import OpenAI, RateLimitError, APIStatusError
    client = OpenAI()
    backoff = 1.0
    for attempt in range(8):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content, {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }
        except RateLimitError as e:
            wait = None
            try:
                msg = str(e)
                m = re.search(r"try again in ([\d.]+)s", msg)
                if m:
                    wait = float(m.group(1)) + 0.5
            except Exception:
                pass
            if wait is None:
                wait = min(backoff, 30.0)
                backoff *= 2.0
            time.sleep(wait)
        except APIStatusError as e:
            if getattr(e, "status_code", None) == 429:
                time.sleep(min(backoff, 30.0))
                backoff *= 2.0
                continue
            raise
    raise RuntimeError("[call_openai] exceeded retries on rate limit")


JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def parse_trace(raw: str) -> dict | None:
    if not raw:
        return None
    if raw.lstrip().startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n", "", raw.strip())
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    m = JSON_BLOCK_RE.search(raw)
    if m:
        blob = m.group(0)
        try:
            return json.loads(blob)
        except Exception:
            try:
                blob2 = re.sub(r",\s*}", "}", blob)
                blob2 = re.sub(r",\s*]", "]", blob2)
                return json.loads(blob2)
            except Exception:
                pass

    fa_obj = _extract_final_answer_fragment(raw)
    if fa_obj is not None:
        return {"final_answer": fa_obj}
    fa_fields = _scrape_final_fields(raw)
    if fa_fields:
        return {"final_answer": fa_fields}
    return None


def _extract_final_answer_fragment(raw: str) -> dict | None:
    idx = raw.find('"final_answer"')
    if idx < 0:
        return None
    open_idx = raw.find("{", idx)
    if open_idx < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(open_idx, len(raw)):
        c = raw[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                blob = raw[open_idx : i + 1]
                try:
                    return json.loads(re.sub(r",\s*}", "}", blob))
                except Exception:
                    return None
    return None


def _scrape_final_fields(raw: str) -> dict | None:
    out: dict = {}
    m = re.search(r'"family"\s*:\s*"([A-Za-z_]+)"', raw)
    if m and m.group(1) in FAMILIES:
        out["family"] = m.group(1)
    if not out:
        return None
    sub = re.search(r'"subtype"\s*:\s*"([^"\n]+)"', raw)
    if sub:
        out["subtype"] = sub.group(1)
    dt = re.search(r'"direction_tag"\s*:\s*"([^"\n]+)"', raw)
    if dt:
        out["direction_tag"] = dt.group(1)
    pol = re.search(r'"polarity"\s*:\s*"([^"\n]+)"', raw)
    if pol:
        out["polarity"] = pol.group(1)
    conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
    if conf:
        try:
            out["confidence"] = float(conf.group(1))
        except Exception:
            pass
    abst = re.search(r'"abstain"\s*:\s*(true|false)', raw)
    if abst:
        out["abstain"] = abst.group(1) == "true"
    return out


def trace_to_final_prediction(trace: dict) -> dict:
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
    ap.add_argument("--pair_ids_file", default=None,
                    help="JSON list of pair_ids to predict (overrides stratified sampling).")
    ap.add_argument("--output", required=True)
    ap.add_argument("--provider", choices=["anthropic", "openai", "google"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--throttle_ms", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--input_order", default="ab")
    args = ap.parse_args()

    manifest = set()
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                manifest.add(json.loads(line)["pair_id"])
    truth = _truth()

    prompts = {}
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("pair_id")
            if pid in manifest:
                prompts[pid] = r["messages"]

    if args.pair_ids_file:
        with open(args.pair_ids_file) as f:
            ids = json.load(f)
        ids = [pid for pid in ids if pid in prompts and pid in truth]
        print(f"[frontier-traces] using {len(ids)} preselected pair_ids from {args.pair_ids_file}")
    else:
        candidate_ids = sorted(set(prompts) & set(truth))
        if args.stratified:
            ids = _stratified_sample(candidate_ids, truth, args.n_pairs, args.seed)
        else:
            rng = random.Random(args.seed)
            ids = candidate_ids[:]
            rng.shuffle(ids)
            ids = ids[: args.n_pairs]

    done = set()
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
    print(f"[frontier-traces] pairs to score: {len(todo)} "
          f"(of {len(ids)} sampled, {len(done)} already done)")

    caller = {
        "anthropic": call_anthropic,
        "openai": call_openai,
        "google": call_google,
    }[args.provider]

    n_ok = n_fail = 0
    total_in_tok = total_out_tok = 0
    t0 = time.time()
    with out_path.open("a") as fout:
        for i, pid in enumerate(todo, 1):
            messages = prompts[pid]
            try:
                raw, usage = caller(messages, args.model, args.max_tokens, args.temperature)
            except Exception as e:
                print(f"[frontier-traces] {pid} call failed: {e}", file=sys.stderr)
                err_rec = {
                    "pair_id": pid,
                    "input_order": args.input_order,
                    "model": f"{args.provider}:{args.model}",
                    "error": str(e)[:500],
                }
                fout.write(json.dumps(err_rec) + "\n")
                fout.flush()
                n_fail += 1
                continue
            trace = parse_trace(raw)
            final = trace_to_final_prediction(trace)
            rec = {
                "pair_id": pid,
                "input_order": args.input_order,
                "model": f"{args.provider}:{args.model}",
                "final_prediction": final,
                "raw_trace_present": trace is not None,
                "raw_output": raw,
                "trace": trace,
                "usage": usage,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            total_in_tok += usage.get("input_tokens", 0)
            total_out_tok += usage.get("output_tokens", 0)
            if trace is None:
                n_fail += 1
            else:
                n_ok += 1
            if i % 10 == 0:
                rate = i / max(time.time() - t0, 1.0)
                print(f"[frontier-traces] {i}/{len(todo)} parse_ok={n_ok} fail={n_fail} "
                      f"in_tok={total_in_tok} out_tok={total_out_tok} ({rate:.2f} rec/s)")
            if args.throttle_ms:
                time.sleep(args.throttle_ms / 1000.0)

    elapsed = time.time() - t0
    print(f"[frontier-traces] done. parse_ok={n_ok} fail={n_fail}; "
          f"in_tok={total_in_tok} out_tok={total_out_tok}; "
          f"elapsed={elapsed/60:.1f}min; wrote {out_path}")


if __name__ == "__main__":
    main()
