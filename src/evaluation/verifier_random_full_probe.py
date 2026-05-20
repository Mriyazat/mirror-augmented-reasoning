"""Probe lightweight verifier policies for improving the DDI student.

This script is intentionally diagnostic rather than a final production router.
It asks:

1. Are simple mechanistic-consistency flags predictive of wrong final answers?
2. Can we choose among pre-SFT / trace-align-v1 / trace-align-v2 predictions
   using only verifier features, without seeing gold labels?
3. What is the upper bound if we repaired only verifier-flagged errors?

Run:
    python -m src.evaluation.verifier_random_full_probe
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/diag2/verifier_probe"
OUT.mkdir(parents=True, exist_ok=True)

FAMS = [
    "AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
    "PK_Distribution", "PK_Excretion", "PK_Metabolism",
]

ENZ_RE = re.compile(r"\b(CYP(?:1A2|2B6|2C8|2C9|2C19|2D6|2E1|3A4|3A5))\b", re.I)
TRANSPORT_RE = re.compile(r"\b(P-?gp|BCRP|OATP1B1|OATP1B3|OAT1|OAT3|OCT2|BSEP|ABCB1|ABCG2)\b", re.I)


def runs_for_split(split: str) -> dict[str, Path]:
    return {
        "pre_sft": ROOT / f"outputs/eval_prompts/pre_sft_greedy_baselines/pred_phase4_{split}_greedy.jsonl",
        "v1": ROOT / f"outputs/student/trace_align/eval_after/pred_traceAlign_{split}_greedy.jsonl",
        "v2": ROOT / f"outputs/student/trace_align/eval_after_v2/pred_traceAlign_v2_{split}_greedy.jsonl",
    }


def load_truth(split: str) -> dict[str, str]:
    truth = {}
    with open(ROOT / f"outputs/eval_prompts/{split}_test_5000_stratified.manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            truth[r["pair_id"]] = r["family"]
    return truth


def load_preds(path: Path, keep: set[str]) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("pair_id")
            if pid in keep and (r.get("input_order") or "ab") == "ab" and pid not in out:
                out[pid] = r
    return out


def final_family(rec: dict) -> str | None:
    fp = rec.get("final_prediction") or {}
    fam = fp.get("family")
    return fam if fam in FAMS else None


def final_conf(rec: dict) -> float:
    fp = rec.get("final_prediction") or {}
    try:
        return float(fp.get("confidence") or 0.0)
    except Exception:
        return 0.0


def final_abstain(rec: dict) -> bool:
    return bool((rec.get("final_prediction") or {}).get("abstain", False))


def trace_obj(rec: dict) -> dict:
    tr = rec.get("trace")
    return tr if isinstance(tr, dict) else {}


def steps(rec: dict) -> list[dict]:
    return trace_obj(rec).get("steps") or []


def all_claims(rec: dict) -> str:
    return "\n".join(str(s.get("claim", "")) for s in steps(rec))


def conclusion_claim(rec: dict) -> str:
    ss = steps(rec)
    if not ss:
        return ""
    for s in reversed(ss):
        if s.get("role") == "conclusion":
            return str(s.get("claim", ""))
    return str(ss[-1].get("claim", ""))


def trace_majority(rec: dict) -> tuple[str | None, float]:
    hints = [
        s.get("family_hint")
        for s in steps(rec)
        if s.get("role") != "conclusion" and s.get("family_hint") in FAMS
    ]
    if not hints:
        return None, 0.0
    fam, n = Counter(hints).most_common(1)[0]
    return fam, n / len(hints)


def cyp_support_score(rec: dict) -> tuple[bool, set[str]]:
    """Return whether trace contains paired metabolism logic for same CYP.

    Conservative: requires one claim with inhibit/induce + enzyme and one
    claim with substrate/metabolized + same enzyme. This catches the common
    error: "A inhibits CYP3A4, therefore affects B" without establishing B is
    a CYP3A4 substrate.
    """
    inhibitor_like: set[str] = set()
    substrate_like: set[str] = set()
    for s in steps(rec):
        claim = str(s.get("claim", ""))
        enzs = {e.upper() for e in ENZ_RE.findall(claim)}
        low = claim.lower()
        if not enzs:
            continue
        if re.search(r"\b(inhibit|inhibits|inhibitor|induce|induces|inducer)\b", low):
            inhibitor_like.update(enzs)
        if re.search(r"\b(substrate|metabolized|metabolised|metabolism of|metabolizes)\b", low):
            substrate_like.update(enzs)
    overlap = inhibitor_like & substrate_like
    return bool(overlap), overlap


def transporter_support_score(rec: dict) -> bool:
    claims = all_claims(rec)
    if not TRANSPORT_RE.search(claims):
        return False
    low = claims.lower()
    return bool(re.search(r"\b(substrate|inhibit|inhibits|inhibitor|transporter|efflux|uptake|excretion|absorption)\b", low))


def verifier_flags(rec: dict) -> dict[str, bool | float | int | str | None]:
    fam = final_family(rec)
    abstain = final_abstain(rec)
    conf = final_conf(rec)
    claims = all_claims(rec)
    concl = conclusion_claim(rec)
    low_claims = claims.lower()
    low_concl = concl.lower()
    tfam, tstrength = trace_majority(rec)
    has_gap = any(s.get("role") == "evidence_gap" for s in steps(rec)) or bool(
        re.search(r"\b(no|insufficient|lack|lacking|without)\b.{0,60}\b(evidence|shared|known|direct|mechanistic|pathway|protein|cyp|interaction)\b", low_claims)
    )
    has_neighbor = any(s.get("role") == "neighbor_pair" for s in steps(rec))
    paired_cyp, cyp_overlap = cyp_support_score(rec)
    has_cyp = bool(ENZ_RE.search(claims))
    has_transporter = transporter_support_score(rec)
    speculative = bool(re.search(r"\b(may|might|could|potentially|possibly|uncertain|insufficient)\b", low_concl))
    invented_neighbor = has_neighbor and bool(re.search(r"analogous pair", low_claims)) and bool(
        re.search(r"no mechanistic-neighbor|no neighbor", low_claims)
    )

    flags = {
        "family": fam,
        "confidence": conf,
        "abstain": abstain,
        "trace_majority": tfam,
        "trace_strength": tstrength,
        "has_gap": has_gap,
        "has_neighbor": has_neighbor,
        "speculative_conclusion": speculative and not abstain,
        "gap_non_abstain": has_gap and not abstain and conf >= 0.35 and not has_neighbor,
        "weak_gap_non_abstain": has_gap and not abstain and conf >= 0.35,
        "pk_metabolism_without_paired_cyp": fam == "PK_Metabolism" and has_cyp and not paired_cyp,
        "pk_nonmetab_without_transport": fam in {"PK_Absorption", "PK_Excretion", "PK_Distribution"} and not has_transporter and not re.search(r"\balbumin|protein.?binding|serum concentration\b", low_claims),
        "adverse_from_gap": fam == "AdverseRisk" and has_gap and not has_neighbor and not abstain,
        "invented_neighbor": invented_neighbor,
        "low_conf_non_abstain": (not abstain) and conf < 0.45,
        "cyp_overlap": ",".join(sorted(cyp_overlap)) if cyp_overlap else None,
    }

    severity = 0.0
    severity += 1.8 if flags["pk_metabolism_without_paired_cyp"] else 0
    severity += 1.5 if flags["adverse_from_gap"] else 0
    severity += 1.2 if flags["gap_non_abstain"] else 0
    severity += 0.8 if flags["speculative_conclusion"] else 0
    severity += 0.8 if flags["invented_neighbor"] else 0
    severity += 0.5 if flags["low_conf_non_abstain"] else 0
    severity -= 0.6 if paired_cyp else 0
    severity -= 0.3 if has_neighbor else 0
    flags["violation_score"] = round(severity, 3)
    return flags


def macro(yt: list[str], yp: list[str]) -> float:
    return float(f1_score(yt, yp, labels=FAMS, average="macro", zero_division=0))


def family_or_default(rec: dict) -> str:
    return final_family(rec) or "PD_Activity"


def choose_by_verifier(cands: dict[str, dict]) -> tuple[str, dict]:
    """Pick the least suspicious candidate; prefer v2 on exact ties."""
    order = {"v2": 0, "v1": 1, "pre_sft": 2}
    scored = []
    for name, rec in cands.items():
        fl = verifier_flags(rec)
        # include confidence only as mild tie breaker; not enough to override hard logic
        score = float(fl["violation_score"]) - 0.08 * final_conf(rec)
        scored.append((score, order.get(name, 99), name, fl))
    scored.sort()
    _, _, name, fl = scored[0]
    return name, fl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random_full", choices=["random_full", "drug_cold", "pair_cold"])
    args = ap.parse_args()

    truth = load_truth(args.split)
    keep = set(truth)
    RUNS = runs_for_split(args.split)
    missing = [str(p) for p in RUNS.values() if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise SystemExit("Missing or empty prediction files:\n  " + "\n  ".join(missing))
    preds = {name: load_preds(path, keep) for name, path in RUNS.items()}
    common = sorted(set.intersection(*(set(p) for p in preds.values())) & keep)
    print(f"[verifier] n_common={len(common)}")

    lines = [f"# Verifier probe on {args.split}\n"]
    lines.append(f"Common pairs across pre-SFT, v1, v2: **{len(common)}**.\n")
    lines.append("\n## Baselines\n")
    lines.append("| Run | Macro-F1 | Accuracy | Mean violation | Flagged high-risk |\n")
    lines.append("|---|---:|---:|---:|---:|\n")

    per_run_flags: dict[str, Counter] = {}
    per_run_err_rates: dict[str, dict] = {}
    for name, src in preds.items():
        yt = [truth[p] for p in common]
        yp = [family_or_default(src[p]) for p in common]
        acc = sum(a == b for a, b in zip(yt, yp)) / len(common)
        flags = [verifier_flags(src[p]) for p in common]
        high = [float(f["violation_score"]) >= 1.5 for f in flags]
        mean_v = float(np.mean([f["violation_score"] for f in flags]))
        lines.append(f"| {name} | {macro(yt, yp):.4f} | {acc:.4f} | {mean_v:.3f} | {sum(high)} ({sum(high)/len(common):.1%}) |\n")

        cnt = Counter()
        err_rates = {}
        for key in [
            "pk_metabolism_without_paired_cyp",
            "adverse_from_gap",
            "gap_non_abstain",
            "weak_gap_non_abstain",
            "speculative_conclusion",
            "invented_neighbor",
            "low_conf_non_abstain",
        ]:
            idx = [i for i, f in enumerate(flags) if f[key]]
            cnt[key] = len(idx)
            if idx:
                err_rates[key] = 1 - sum(yp[i] == yt[i] for i in idx) / len(idx)
            else:
                err_rates[key] = None
        per_run_flags[name] = cnt
        per_run_err_rates[name] = err_rates

    lines.append("\n## Flag prevalence and error rate\n")
    for name in preds:
        lines.append(f"\n### {name}\n")
        lines.append("| Flag | Count | Error rate among flagged |\n")
        lines.append("|---|---:|---:|\n")
        for key, count in per_run_flags[name].most_common():
            er = per_run_err_rates[name][key]
            er_s = "—" if er is None else f"{er:.1%}"
            lines.append(f"| {key} | {count} | {er_s} |\n")

    # Candidate selection among the three student variants
    yt = [truth[p] for p in common]
    chosen_names = []
    chosen_flags = []
    chosen_pred = []
    for p in common:
        cands = {name: preds[name][p] for name in preds}
        cname, cflags = choose_by_verifier(cands)
        chosen_names.append(cname)
        chosen_flags.append(cflags)
        chosen_pred.append(family_or_default(preds[cname][p]))
    lines.append("\n## Verifier-rerank among pre-SFT / v1 / v2\n")
    lines.append(f"- Macro-F1: **{macro(yt, chosen_pred):.4f}**\n")
    lines.append(f"- Accuracy: **{sum(a == b for a, b in zip(yt, chosen_pred)) / len(yt):.4f}**\n")
    lines.append("- Chosen source distribution: " + ", ".join(f"`{k}`={v}" for k, v in Counter(chosen_names).most_common()) + "\n")

    # Oracle: if any candidate is right, upper bound
    oracle = []
    for p in common:
        gold = truth[p]
        candidates = [family_or_default(preds[name][p]) for name in preds]
        oracle.append(gold if gold in candidates else candidates[0])
    lines.append(f"- Oracle among 3 runs: **{macro(yt, oracle):.4f}** macro-F1 (upper bound for reranking only)\n")

    # Selective risk: if we abstain high violation score, what is retained accuracy?
    lines.append("\n## Selective verifier risk curves (v2)\n")
    lines.append("| Threshold: abstain if violation ≥ | Coverage | Accuracy retained | Error rate flagged |\n")
    lines.append("|---:|---:|---:|---:|\n")
    v2 = preds["v2"]
    v2_yp = [family_or_default(v2[p]) for p in common]
    v2_flags = [verifier_flags(v2[p]) for p in common]
    for th in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        keep_idx = [i for i, f in enumerate(v2_flags) if float(f["violation_score"]) < th]
        flag_idx = [i for i, f in enumerate(v2_flags) if float(f["violation_score"]) >= th]
        cov = len(keep_idx) / len(common)
        acc_keep = sum(v2_yp[i] == yt[i] for i in keep_idx) / max(1, len(keep_idx))
        err_flag = 1 - sum(v2_yp[i] == yt[i] for i in flag_idx) / max(1, len(flag_idx))
        lines.append(f"| {th:.1f} | {cov:.1%} | {acc_keep:.1%} | {err_flag:.1%} |\n")

    # Examples for repair dataset
    lines.append("\n## High-confidence repair candidates from v2\n")
    lines.append("These are cases where v2 is wrong and the verifier gives a high violation score; they are good candidates for a repair-SFT dataset.\n\n")
    lines.append("| pair_id | gold | pred | violation | trace_majority | conf | key reason |\n")
    lines.append("|---|---|---|---:|---|---:|---|\n")
    shown = 0
    for p, f, pred in sorted(
        zip(common, v2_flags, v2_yp), key=lambda x: -float(x[1]["violation_score"])
    ):
        if pred == truth[p] or shown >= 30:
            continue
        reasons = [
            k for k in [
                "pk_metabolism_without_paired_cyp",
                "adverse_from_gap",
                "gap_non_abstain",
                "speculative_conclusion",
                "invented_neighbor",
            ]
            if f[k]
        ]
        lines.append(
            f"| `{p}` | {truth[p]} | {pred} | {f['violation_score']:.1f} | "
            f"{f['trace_majority']} ({f['trace_strength']:.2f}) | {f['confidence']:.2f} | "
            f"{', '.join(reasons)} |\n"
        )
        shown += 1

    out_path = OUT / f"{args.split}_verifier_probe.md"
    out_path.write_text("".join(lines))
    print(f"[verifier] wrote {out_path}")
    print("".join(lines[:30]))


if __name__ == "__main__":
    main()
