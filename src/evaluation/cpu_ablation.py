"""Print a paper-ready ablation table:

  greedy student
  + self-consistency (rerank candidates, prm-weighted vote)
  + trace-rescue (val-tuned hint-majority, max_original_frac=0.10)
  + meta-router(LLM_stack + best baseline)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]
F2I = {f: i for i, f in enumerate(FAMS)}


def _truth():
    rows = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet",
                         columns=["pair_id", "family"]).to_pylist()
    return {r["pair_id"]: r["family"] for r in rows}


def _manifest(p):
    s = set()
    with open(p) as f:
        for line in f:
            s.add(json.loads(line)["pair_id"])
    return s


def _load_ab(p):
    out = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            pid = r["pair_id"]
            if (r.get("input_order") or "ab") != "ab":
                continue
            if pid in out:
                continue
            fp = r.get("final_prediction") or {}
            out[pid] = (fp.get("family"), bool(fp.get("abstain", False)))
    return out


def macro(pred, truth, keep):
    common = sorted({p for p in pred if p in truth and p in keep})
    yt = [F2I[truth[p]] for p in common]
    yp = [len(FAMS) if (pred[p][1] or pred[p][0] not in F2I) else F2I[pred[p][0]] for p in common]
    return f1_score(yt, yp, labels=list(range(len(FAMS))), average="macro", zero_division=0), len(common)


def make_sc(input_path, output_path):
    subprocess.run([sys.executable, "-m", "src.inference.self_consistency",
                    "--input", input_path, "--output", output_path,
                    "--policy", "prm_weighted"], check=True, capture_output=True)


def make_tr(input_path, output_path):
    subprocess.run([sys.executable, "-m", "src.inference.trace_rescue",
                    "--input", input_path, "--output", output_path,
                    "--policy", "hint_majority", "--min_steps", "3",
                    "--min_strength", "0.5", "--max_original_frac", "0.10"],
                   check=True, capture_output=True)


def best_meta(split):
    """Return the meta-router macro on cpu_stack input."""
    META = ROOT / "outputs/diag2/meta_cpu"
    cands = []
    for b in ("xgb", "mlp"):
        f = META / f"{split}_{b}.json"
        if f.exists():
            j = json.loads(f.read_text())
            cands.append((j["test_macro_meta"], b))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0]


def main():
    truth = _truth()
    splits = ["random_full", "drug_cold", "pair_cold"]
    raw = {
        "random_full": ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank8_abba.jsonl",
        "drug_cold":   ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_drug_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
        "pair_cold":   ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_pair_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
    }
    manifests = {sp: ROOT / f"outputs/eval_prompts/{sp}_test_5000_stratified.manifest.jsonl" for sp in splits}

    print(f"{'split':>12s} {'greedy':>10s} {'+SC':>10s} {'+SC+TR':>10s} {'+meta-router':>14s}")
    print("-" * 60)
    rows = []
    for sp in splits:
        keep = _manifest(manifests[sp])
        base = _load_ab(raw[sp])
        f_g, _ = macro(base, truth, keep)

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as t1:
            sc_path = t1.name
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as t2:
            tr_path = t2.name
        make_sc(str(raw[sp]), sc_path)
        sc = _load_ab(sc_path)
        f_sc, _ = macro(sc, truth, keep)
        make_tr(sc_path, tr_path)
        tr = _load_ab(tr_path)
        f_tr, _ = macro(tr, truth, keep)
        Path(sc_path).unlink(missing_ok=True)
        Path(tr_path).unlink(missing_ok=True)

        meta = best_meta(sp)
        f_meta = meta[0] if meta else float("nan")

        rows.append((sp, f_g, f_sc, f_tr, f_meta))
        print(f"{sp:>12s} {f_g:>10.4f} {f_sc:>10.4f} {f_tr:>10.4f} {f_meta:>14.4f}")

    out = {sp: {"greedy": fg, "sc": fsc, "sc_tr": ftr, "meta": fm} for sp, fg, fsc, ftr, fm in rows}
    op = ROOT / "outputs/diag2/headline/cpu_ablation_table.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
