"""A0.6 — Label co-occurrence audit on new DrugBank (2026-04).

Goal: For every (drug_A, drug_B) pair appearing in <drug-interactions>, count
how many DISTINCT interaction <description> strings are attached across the
whole DrugBank. Also record per-description directionality (A->B vs B->A).

Why: V3 assumed 1 pair -> 1 label. If DrugBank lists multiple descriptions per
pair, V4 must define an explicit resolution policy (e.g., union of mechanisms,
canonical directionality, or multi-label).

Outputs (DDI/outputs/audit/):
  a06_pair_label_counts.jsonl   # {pair: "A|B", n_descriptions, descriptions:[...], directions:[...]}
  a06_summary.json              # aggregate stats
  a06_report.md                 # human-readable findings

Runs in streaming (iterparse) mode. ~5-10 min on 2.4GB.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT / "data_raw" / "drugbank_2026-04.xml"
OUT = ROOT / "outputs" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

NS = "{http://www.drugbank.ca}"
DRUG = f"{NS}drug"
INTERACTIONS = f"{NS}drug-interactions"
INTERACTION = f"{NS}drug-interaction"
DB_ID_TAG = f"{NS}drugbank-id"
NAME_TAG = f"{NS}name"
DESC_TAG = f"{NS}description"


def primary_drugbank_id(drug_elem: ET.Element) -> str | None:
    """Return the primary DrugBank ID (attr primary="true", else first)."""
    first = None
    for dbid in drug_elem.findall(DB_ID_TAG):
        if first is None:
            first = dbid.text
        if dbid.get("primary") == "true":
            return dbid.text
    return first


def main() -> None:
    assert XML.exists(), f"missing {XML}"

    t0 = time.time()
    print(f"[audit] parsing {XML} ({XML.stat().st_size/1e9:.2f} GB)", flush=True)

    # pair_desc[(a,b_sorted)] = {"descs": {desc: set(directions)}, "names":{a:"..",b:".."}}
    pair_desc: dict[tuple[str, str], dict] = defaultdict(lambda: {"descs": defaultdict(set)})
    drug_names: dict[str, str] = {}

    n_drugs = 0
    n_interactions = 0
    depth = 0

    ctx = ET.iterparse(str(XML), events=("start", "end"))
    current_drug_id: str | None = None
    current_drug_name: str | None = None

    for event, elem in ctx:
        if event == "start":
            if elem.tag == DRUG:
                depth += 1
                if depth == 1:
                    current_drug_id = None
                    current_drug_name = None
        else:  # end
            if elem.tag == DRUG and depth == 1:
                n_drugs += 1
                if current_drug_id:
                    drug_names[current_drug_id] = current_drug_name or current_drug_id
                if n_drugs % 2000 == 0:
                    print(
                        f"  parsed {n_drugs} drugs | {n_interactions} interactions "
                        f"| {len(pair_desc)} pairs | {time.time()-t0:.0f}s",
                        flush=True,
                    )
                elem.clear()
                depth -= 1
                current_drug_id = None
                current_drug_name = None
                continue
            if elem.tag == DRUG:
                depth -= 1
                continue

            # Only process top-level drug metadata
            if depth != 1:
                continue

            if elem.tag == DB_ID_TAG and current_drug_id is None:
                # First drugbank-id within a top-level drug is primary (they are listed first)
                if elem.get("primary") == "true" or current_drug_id is None:
                    current_drug_id = elem.text
            elif elem.tag == NAME_TAG and current_drug_name is None:
                current_drug_name = elem.text
            elif elem.tag == INTERACTION:
                # child elements: drugbank-id, name, description
                b_id = None
                b_name = None
                desc = None
                for c in elem:
                    if c.tag == DB_ID_TAG and b_id is None:
                        b_id = c.text
                    elif c.tag == NAME_TAG and b_name is None:
                        b_name = c.text
                    elif c.tag == DESC_TAG:
                        desc = c.text
                if current_drug_id and b_id and desc:
                    n_interactions += 1
                    if b_name and b_id not in drug_names:
                        drug_names[b_id] = b_name
                    a = current_drug_id
                    # Canonical pair (sorted) + record direction
                    if a < b_id:
                        pair = (a, b_id)
                        direction = f"{a}->{b_id}"
                    else:
                        pair = (b_id, a)
                        direction = f"{a}->{b_id}"  # preserve original
                    pair_desc[pair]["descs"][desc].add(direction)
                elem.clear()

    print(
        f"[audit] done parse: {n_drugs} drugs, {n_interactions} interactions, "
        f"{len(pair_desc)} unique pairs, {time.time()-t0:.0f}s",
        flush=True,
    )

    # Write per-pair JSONL
    pair_counts_path = OUT / "a06_pair_label_counts.jsonl"
    multi_desc_counts = 0
    bidirectional_asymmetric = 0  # pair with both A->B and B->A but different desc sets
    desc_counter = defaultdict(int)
    pair_ndesc_dist = defaultdict(int)
    with pair_counts_path.open("w") as f:
        for (a, b), rec in pair_desc.items():
            descs = rec["descs"]
            n = len(descs)
            pair_ndesc_dist[n] += 1
            if n > 1:
                multi_desc_counts += 1
            # directionality check
            dir_map: dict[str, set[str]] = defaultdict(set)  # direction -> set of desc
            for d, dirs in descs.items():
                for dr in dirs:
                    dir_map[dr].add(d)
            directions = sorted(dir_map.keys())
            if len(directions) == 2:
                if dir_map[directions[0]] != dir_map[directions[1]]:
                    bidirectional_asymmetric += 1
            desc_counter[n] += 1
            out = {
                "pair": f"{a}|{b}",
                "n_descriptions": n,
                "descriptions": sorted(descs.keys()),
                "directions_per_description": {d: sorted(dirs) for d, dirs in descs.items()},
                "directions_observed": directions,
            }
            f.write(json.dumps(out) + "\n")

    # Summary
    total_pairs = len(pair_desc)
    summary = {
        "drugbank_file": str(XML),
        "n_drugs_parsed": n_drugs,
        "n_interactions": n_interactions,
        "n_unique_pairs": total_pairs,
        "n_pairs_with_multiple_descriptions": multi_desc_counts,
        "pct_pairs_multi_desc": round(100.0 * multi_desc_counts / max(total_pairs, 1), 3),
        "n_bidirectional_asymmetric_pairs": bidirectional_asymmetric,
        "pct_bidirectional_asymmetric": round(
            100.0 * bidirectional_asymmetric / max(total_pairs, 1), 3
        ),
        "pair_ndesc_distribution": dict(sorted(pair_ndesc_dist.items())),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "a06_summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown report
    md = []
    md.append("# A0.6 — Label Co-occurrence Audit on DrugBank 2026-04\n")
    md.append(f"- DrugBank file: `{XML}`")
    md.append(f"- Drugs parsed: **{n_drugs}**")
    md.append(f"- Total drug-interaction rows: **{n_interactions}**")
    md.append(f"- Unique unordered pairs: **{total_pairs}**")
    md.append(f"- Pairs with MORE than one distinct description: **{multi_desc_counts}** "
              f"({summary['pct_pairs_multi_desc']}%)")
    md.append(f"- Pairs with asymmetric description between directions: "
              f"**{bidirectional_asymmetric}** ({summary['pct_bidirectional_asymmetric']}%)")
    md.append("\n## Distribution: #descriptions per pair")
    md.append("| n_descriptions | n_pairs |")
    md.append("|---:|---:|")
    for k, v in summary["pair_ndesc_distribution"].items():
        md.append(f"| {k} | {v} |")
    md.append("\n## V4 Resolution Policy Options")
    md.append("Based on the counts above, we will decide in A0.10 pre-flight report:")
    md.append("1. **single-description** (drop pairs with conflicts) — simplest, lose coverage")
    md.append("2. **canonical-first** (pick primary DrugBank direction description) — matches biology of \"subject affects object\"")
    md.append("3. **multi-label** (allow pair -> {label_1, ..., label_k}) — preserves info but changes task")
    md.append("4. **union-text** (concatenate descriptions into one canonical string before label mapping) — preserves info, keeps single-label")
    md.append(f"\n_Parse time: {summary['elapsed_sec']}s._")
    (OUT / "a06_report.md").write_text("\n".join(md) + "\n")

    print(f"[audit] wrote {pair_counts_path}")
    print(f"[audit] wrote {OUT/'a06_summary.json'}")
    print(f"[audit] wrote {OUT/'a06_report.md'}")


if __name__ == "__main__":
    main()
