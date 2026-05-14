"""B1b — Teacher trace generator.

Iterates a split manifest, builds a context+prompt for each pair, asks the
teacher LLM for N candidates, and writes raw candidates to a JSONL file:
  outputs/teacher/raw_<split>_<provider>.jsonl

Each line = one candidate:
    {
      "pair_id":      "DB00091|DB00602",
      "candidate_id": 0,                     # 0..N-1
      "provider":     "openai:vllm/Llama-3.3-70B-Instruct",
      "temperature":  0.8,
      "seed":         12345,
      "latency_ms":   4510,
      "prompt_tokens":   1420,
      "completion_tokens": 310,
      "raw_text":     "<model output>",      # the text; B2 parses JSON from this
      "gold_family":  "PK_Metabolism",       # for eval/analysis only; not in prompt
      "gold_direction": "a_to_b"
    }

Design:
  - Safe to resume: we skip pairs whose pair_id is already present in the
    output JSONL at startup.
  - Every prompt is saved on-disk under outputs/teacher/prompts_<split>/<pair>.txt
    (optional, --save_prompts) so a reviewer can audit the exact input.
  - Fails soft: one bad pair doesn't kill the run; errors go to
    outputs/teacher/gen_errors_<split>_<provider>.jsonl.

Usage:
  # Local dev: N=1, N=5 prototype
  python -m src.teacher.generate --split subset25k --provider ollama \\
       --model llama3.1:8b --limit 100 --n 3

  # On-cluster vLLM (4×H100 Llama-3.3-70B):
  python -m src.teacher.generate --split subset25k --provider openai \\
       --model meta-llama/Llama-3.3-70B-Instruct \\
       --base-url http://localhost:8000/v1 --n 5

  # Dummy (CI smoke test; no LLM):
  python -m src.teacher.generate --split subset25k --provider dummy \\
       --limit 20 --n 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from src.teacher.context_builder import ContextBuilder, DATA
from src.teacher.prompt import build_prompt
from src.teacher.provider import make_provider, Generation
from src.teacher.schema import load_rubric

ROOT = Path(__file__).resolve().parents[2]
# Teacher writes GB of JSONL per run — route to $DDI_OUTPUTS (set by
# activate_env.sh to $SCRATCH/ddi_v4_outputs on the cluster) when present,
# falling back to ./outputs for local dev.
OUT_DIR = Path(os.environ.get("DDI_OUTPUTS", ROOT / "outputs")) / "teacher"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _out_paths(split: str, provider_tag: str):
    raw = OUT_DIR / f"raw_{split}_{provider_tag}.jsonl"
    errs = OUT_DIR / f"gen_errors_{split}_{provider_tag}.jsonl"
    prompts_dir = OUT_DIR / f"prompts_{split}"
    return raw, errs, prompts_dir


def _model_tag(model: str) -> str:
    """Filesystem-safe compact tag for a model name."""
    return model.replace("/", "_").replace(":", "_")


def _resolve_teacher(model: str) -> tuple[str, str | None]:
    """Map a HF model path back to its rubric (teacher_id, teacher_family).

    The rubric is the single source of truth (configs/prm_rubric.yaml ->
    teacher_models[*]).  Falling back to (_model_tag, None) preserves
    backwards-compatibility with arbitrary --model values used during
    smoke tests, while production runs that match a rubric entry get
    both id and family populated automatically — no extra CLI flags
    required from run_teacher.sh.
    """
    try:
        rubric = load_rubric()
        for entry in rubric.get("teacher_models") or []:
            if entry.get("path") == model:
                return entry.get("id") or _model_tag(model), entry.get("family")
    except Exception:
        pass  # rubric missing/malformed -> fall through to defaults
    return _model_tag(model), None


def _resume_cleanup(path: Path, n_expected: int) -> tuple[set[str], int, int]:
    """Robust resume: a pair is 'done' ONLY if it has >= n_expected candidate
    records in the output file. Records belonging to pairs with fewer than
    n_expected candidates (SIGTERM'd mid-pair) are removed from the file
    so resume can cleanly re-generate them without duplicates.

    Returns (done_pair_ids, n_complete_pairs, n_removed_records).
    """
    if not path.exists():
        return set(), 0, 0
    buckets: dict[str, list[str]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                buckets[r["pair_id"]].append(line)
            except Exception:
                pass  # skip malformed tail from SIGKILL
    done = {pid for pid, lines in buckets.items() if len(lines) >= n_expected}
    n_removed = sum(len(lines) for pid, lines in buckets.items() if pid not in done)
    if n_removed > 0:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            for pid in done:
                for line in buckets[pid]:
                    f.write(line)
        tmp.replace(path)  # atomic on POSIX
    return done, len(done), n_removed


# SIGTERM trap for SLURM preemption: finish the current pair atomically
# then exit clean. SLURM sends SIGTERM ~30-90 seconds before SIGKILL.
_STOP_REQUESTED = False


def _install_sigterm_handler():
    def _handler(signum, frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        print(f"\n[gen] received signal {signum}; will exit after current pair",
              flush=True)

    signal.signal(signal.SIGTERM, _handler)
    try:
        signal.signal(signal.SIGUSR1, _handler)  # SLURM --signal=USR1@120
    except Exception:
        pass


def _labels_lookup() -> dict[str, dict]:
    tbl = pq.read_table(DATA / "labels_hierarchical.parquet",
                        columns=["pair_id", "family", "subtype",
                                 "subject_drugbank_id", "object_drugbank_id",
                                 "bidirectional", "polarity", "a_id", "b_id"]
                        ).to_pylist()
    labels: dict[str, dict] = {}
    for r in tbl:
        if r.get("bidirectional"):
            dt = "bidirectional"
        elif r["subject_drugbank_id"] == r["a_id"]:
            dt = "a_to_b"
        elif r["subject_drugbank_id"] == r["b_id"]:
            dt = "b_to_a"
        else:
            dt = "n/a"
        labels[r["pair_id"]] = {
            "family": r["family"],
            "subtype": r["subtype"],
            "direction_tag": dt,
            "polarity": r.get("polarity"),
        }
    return labels


def _run_single(split: str, provider_name: str, model: str,
                n: int, manifest: list[dict], labels: dict,
                cb: ContextBuilder, temperature: float, top_p: float,
                max_tokens: int, seed: int, base_url: str | None,
                save_prompts: bool, resume: bool,
                tensor_parallel_size: int,
                teacher_id: str | None = None,
                teacher_family: str | None = None,
                guidance_by_pair: dict[str, str] | None = None,
                prompt_version_suffix: str | None = None) -> None:
    """Run generation with ONE teacher model and write to its own JSONL."""
    provider_tag = f"{provider_name}-{_model_tag(model)}"
    raw_path, errs_path, prompts_dir = _out_paths(split, provider_tag)
    if save_prompts:
        prompts_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[gen] === teacher: {teacher_id or model} ===")
    print(f"[gen]   provider={provider_name}  model={model}")
    print(f"[gen]   n={n}  temperature={temperature}  top_p={top_p}  "
          f"max_tokens={max_tokens}  seed={seed}")
    try:
        _rel = str(raw_path.relative_to(ROOT))
    except ValueError:
        _rel = str(raw_path)  # on-scratch path, print absolute
    print(f"[gen]   output: {_rel}")

    prov = make_provider(
        provider_name, model=model, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, base_url=base_url,
        tensor_parallel_size=tensor_parallel_size,
    )

    # Temperature schedule for candidate diversification (v4.3).
    rubric = load_rubric()
    temp_schedule: list[float] = list(
        rubric.get("teacher_generation", {}).get("candidate_temperatures") or []
    )
    if temp_schedule:
        print(f"[gen]   temp_schedule: {temp_schedule[:n]}")

    if resume:
        done, n_done, n_removed = _resume_cleanup(raw_path, n_expected=n)
        if n_done or n_removed:
            print(f"[gen]   resume: {n_done:,} complete pairs kept; "
                  f"{n_removed:,} partial records from incomplete pairs removed")
    else:
        done = set()

    _install_sigterm_handler()

    # Filter out already-done pairs up front so the context prefetch pipeline
    # doesn't waste work building contexts we'll throw away. Preserve the
    # original manifest index -- it feeds into the per-candidate seed.
    work_items = [(i, row) for i, row in enumerate(manifest)
                  if row["pair_id"] not in done]

    # Context prefetch: ContextBuilder.build() is ~10-17s (parquet neighbor
    # reads + RAG retrieval + prompt assembly).  Without prefetch the 4 GPUs
    # sit idle for that window between pairs.  One-ahead prefetch overlaps
    # ctx-build with the 24-way generation, ~halving observed per-pair time
    # (confirmed: gen ~17s, ctx ~17s, so pipelined ≈ max(gen, ctx) ≈ 17s).
    ctx_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ctx")

    def _prefetch_ctx(work_k: int):
        if work_k >= len(work_items):
            return None
        _, wrow = work_items[work_k]
        try:
            return ("ok", cb.build(wrow["pair_id"]))
        except Exception as _e:
            return ("err", str(_e))

    next_ctx_fut = ctx_pool.submit(_prefetch_ctx, 0) if work_items else None

    # Persistent gen pool — one thread per candidate slot. Reused across pairs
    # so we don't pay executor-setup cost each pair (shaves ~50ms/pair).
    gen_pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="gen")

    t0 = time.time()
    n_written = 0
    n_err = 0
    with raw_path.open("a") as fout, errs_path.open("a") as ferr:
        for k, (idx, row) in enumerate(work_items):
            if _STOP_REQUESTED:
                print(f"[gen]   stopping cleanly at pair {k}/{len(work_items)}")
                break
            pid = row["pair_id"]
            ctx_result = next_ctx_fut.result() if next_ctx_fut else None
            # Kick off prefetch of k+1 BEFORE we start this pair's 24-way gen
            # so cb.build() runs in parallel with the GPU work.
            next_ctx_fut = ctx_pool.submit(_prefetch_ctx, k + 1)

            if ctx_result is None or ctx_result[0] == "err":
                err_msg = ctx_result[1] if ctx_result else "empty context"
                ferr.write(json.dumps({"pair_id": pid, "stage": "context",
                                       "error": err_msg}) + "\n")
                n_err += 1
                continue
            ctx = ctx_result[1]
            extra_guidance = (
                guidance_by_pair.get(pid) if guidance_by_pair else None
            )
            msgs = build_prompt(
                ctx,
                extra_user_guidance=extra_guidance,
                prompt_version_suffix=prompt_version_suffix,
            )
            if save_prompts:
                (prompts_dir / f"{pid.replace('|', '_')}.txt").write_text(
                    f"### SYSTEM\n{msgs['system']}\n\n### USER\n{msgs['user']}\n"
                )

            # Temperature diversification (v4.3): N candidates with different
            # temperatures for reasoning diversity (Feng et al. 2024).  Fan
            # out the N calls concurrently so vLLM can batch them in a single
            # forward pass instead of serving them one at a time.  On a 4×H100
            # node this cuts per-pair latency from ~5 min (sequential) to
            # ~15-20 s (batched).
            per_call_temps: list[float] = [
                temp_schedule[ci % len(temp_schedule)]
                if temp_schedule else prov.temperature
                for ci in range(n)
            ]
            try:
                def _call_one(ci: int) -> Generation:
                    sub = prov.generate(
                        system=msgs["system"], user=msgs["user"],
                        n=1, seed=seed + idx * 100 + ci,
                        temperature_override=per_call_temps[ci],
                    )
                    return sub[0]

                # preserve candidate_id order (0..n-1) regardless of
                # completion order -> reproducibility of seed+ci offset
                gens: list[Generation] = list(gen_pool.map(_call_one, range(n)))
            except Exception as e:
                ferr.write(json.dumps({"pair_id": pid, "stage": "generate",
                                       "error": str(e)}) + "\n")
                ferr.flush()
                n_err += 1
                continue

            gold = labels.get(pid, {})
            # Atomic per-pair write: buffer all N candidate records as one
            # string, then single write + flush + fsync. Keeps the file
            # well-formed even if SIGKILL lands mid-pair.
            pair_lines: list[str] = []
            for ci, g in enumerate(gens):
                rec = {
                    "pair_id":     pid,
                    "candidate_id": ci,
                    "teacher_id":    teacher_id or _model_tag(model),
                    "teacher_family": teacher_family,
                    "provider":    f"{prov.name}:{prov.model}",
                    "temperature": per_call_temps[ci] if ci < len(per_call_temps) else prov.temperature,
                    "top_p":       prov.top_p,
                    "seed":        seed + idx,
                    "prompt_version": msgs.get("prompt_version"),
                    "prompt_sha":     msgs.get("prompt_sha"),
                    "latency_ms":  g.latency_ms,
                    "prompt_tokens":     g.prompt_tokens,
                    "completion_tokens": g.completion_tokens,
                    "finish_reason":     g.finish_reason,
                    "raw_text":    g.text,
                    "gold_family":    gold.get("family"),
                    "gold_subtype":   gold.get("subtype"),
                    "gold_direction": gold.get("direction_tag"),
                    "gold_polarity":  gold.get("polarity"),
                }
                pair_lines.append(json.dumps(rec) + "\n")
                n_written += 1
            fout.write("".join(pair_lines))
            fout.flush()
            try:
                os.fsync(fout.fileno())  # force page-cache → disk
            except OSError:
                pass  # Lustre/NFS may not support full fsync

            if (k + 1) % 50 == 0:
                rate = (k + 1) / max(1, time.time() - t0)
                eta = (len(work_items) - k - 1) / max(1e-9, rate)
                print(f"[gen]   {k+1:,}/{len(work_items):,} (of "
                      f"{len(manifest):,} total)  "
                      f"written={n_written:,}  err={n_err}  "
                      f"{rate:.2f} pair/s  eta={eta:.0f}s", flush=True)

    # Shut down pools cleanly so Python doesn't deadlock on exit.
    gen_pool.shutdown(wait=False, cancel_futures=True)
    ctx_pool.shutdown(wait=False, cancel_futures=True)

    print(f"\n[gen]   teacher {teacher_id or model}: wrote {n_written:,} "
          f"candidates across {len(work_items) - n_err:,} pairs ({n_err} errors)")


def _load_manifest_file(path: str | None) -> list[dict] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest_file does not exist: {p}")
    rows: list[dict] = []
    if p.suffix == ".parquet":
        rows = pq.read_table(p).to_pylist()
    else:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    rows.append(json.loads(line))
                else:
                    rows.append({"pair_id": line.split(",")[0].strip()})
    for r in rows:
        if not r.get("pair_id"):
            raise ValueError(f"manifest_file row missing pair_id: {r}")
    return rows


def _load_guidance_file(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"guidance_file does not exist: {p}")
    out: dict[str, str] = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("pair_id")
            guidance = r.get("guidance")
            if pid and guidance:
                out[pid] = guidance
    return out


def run(split: str, provider: str, model: str,
        n: int, limit: int | None, temperature: float, top_p: float,
        max_tokens: int, seed: int, base_url: str | None,
        save_prompts: bool, resume: bool,
        tensor_parallel_size: int = 4,
        manifest_file: str | None = None,
        guidance_file: str | None = None,
        prompt_version_suffix: str | None = None) -> None:
    """Backwards-compatible single-model driver (multi-teacher uses run_ensemble)."""
    nb_idx = DATA / f"neighbor_index_{split}.parquet"
    cb = ContextBuilder(neighbor_index_path=nb_idx if nb_idx.exists() else None)
    labels = _labels_lookup()
    manifest = _load_manifest_file(manifest_file)
    if manifest is None:
        manifest = pq.read_table(DATA / "splits" / f"manifest_{split}.parquet").to_pylist()
    if limit:
        manifest = manifest[:limit]
    guidance_by_pair = _load_guidance_file(guidance_file)
    if guidance_by_pair:
        missing_guidance = sum(1 for r in manifest if r["pair_id"] not in guidance_by_pair)
        print(f"[gen] loaded guidance for {len(guidance_by_pair):,} pairs; "
              f"{missing_guidance:,}/{len(manifest):,} manifest pairs without guidance")
    print(f"[gen] processing {len(manifest):,} pairs (single-model mode)")

    # Support comma-separated --model for ensemble from CLI; fall back to
    # single-model run for a literal single name.
    models = [m.strip() for m in model.split(",") if m.strip()]
    if len(models) == 1:
        tid, tfam = _resolve_teacher(models[0])
        _run_single(split, provider, models[0], n, manifest, labels, cb,
                    temperature, top_p, max_tokens, seed, base_url,
                    save_prompts, resume, tensor_parallel_size,
                    teacher_id=tid, teacher_family=tfam,
                    guidance_by_pair=guidance_by_pair,
                    prompt_version_suffix=prompt_version_suffix)
    else:
        for midx, m in enumerate(models):
            tid, tfam = _resolve_teacher(m)
            _run_single(split, provider, m, n, manifest, labels, cb,
                        temperature, top_p, max_tokens,
                        seed + midx * 10_000,     # decorrelate seeds per teacher
                        base_url, save_prompts, resume, tensor_parallel_size,
                        teacher_id=tid, teacher_family=tfam,
                        guidance_by_pair=guidance_by_pair,
                        prompt_version_suffix=prompt_version_suffix)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="subset25k")
    p.add_argument("--provider", required=True,
                   choices=["dummy", "ollama", "openai", "vllm"])
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=24,
                   help="Candidates per pair per teacher. "
                        "Rubric default: 24 (3-teacher ensemble → 72/pair). "
                        "Override to 32 for more PRM headroom, or 8 for "
                        "laptop smoke tests.")
    p.add_argument("--limit", type=int)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--manifest-file", default=None,
                   help="Optional JSONL/TXT/CSV/Parquet manifest with pair_id rows. "
                        "Used for targeted regeneration without adding a split.")
    p.add_argument("--guidance-file", default=None,
                   help="Optional JSONL with {pair_id, guidance} to append to "
                        "each pair's teacher prompt.")
    p.add_argument("--prompt-version-suffix", default=None,
                   help="Suffix appended to prompt_version when guidance is used, "
                        "e.g. phase4disambig.")
    p.add_argument("--save-prompts", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    args = p.parse_args()
    run(**vars(args))
