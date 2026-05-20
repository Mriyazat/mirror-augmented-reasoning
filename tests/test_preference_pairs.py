"""Unit tests for src/teacher/build_preference_pairs.py.

Synthesizes a tiny teacher_clean.jsonl with one record per tier/direction
combination, runs the builder, and verifies the expected preference
types appear with the expected direction-flip behavior.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mk_teacher_record(pair_id: str, family: str, subtype: str, direction: str,
                       tier: str, with_evidence: bool = True) -> dict:
    steps = [
        {"step_id": 1, "role": "pathway",
         "claim": "Shared CYP3A4 metabolism pathway.",
         "evidence_ids": ["enzyme_CYP3A4"] if with_evidence else [],
         "direction_tag": "n/a"},
        {"step_id": 2, "role": "pk_flag",
         "claim": "Drug X is a CYP3A4 inhibitor per DrugBank.",
         "evidence_ids": ["pk_flag_cyp3a4_inhibitor"] if with_evidence else [],
         "direction_tag": "n/a"},
        {"step_id": 3, "role": "direction",
         "claim": "Drug X inhibits metabolism of Drug Y.",
         "evidence_ids": [],
         "direction_tag": direction},
        {"step_id": 4, "role": "conclusion",
         "claim": f"This is a {family}/{subtype} interaction.",
         "evidence_ids": [],
         "direction_tag": direction},
    ]
    trace = {
        "steps": steps,
        "final_answer": {
            "family":        family,
            "subtype":       subtype,
            "direction_tag": direction,
            "polarity":      "up",
            "abstain":       False,
        },
    }
    return {
        "pair_id":       pair_id,
        "family":        family,
        "subtype":       subtype,
        "direction_tag": direction,
        "polarity":      "up",
        "messages": [
            {"role": "system", "content": "<sys>"},
            {"role": "user",   "content": f"Pair: {pair_id}"},
            {"role": "assistant", "content": json.dumps(trace)},
        ],
        "tier":           tier,
        "sample_weight":  1.0,
        "critic_score":   0.8,
    }


def test_preference_builder():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        teacher = td / "teacher_clean.jsonl"
        out = td / "prefs.jsonl"
        report = td / "report.md"

        records = [
            _mk_teacher_record("P001", "PK_Metabolism", "inhibits_CYP3A4",
                               "a_to_b", "full_correct"),
            _mk_teacher_record("P002", "PK_Metabolism", "inhibits_CYP3A4",
                               "bidirectional", "full_correct"),
            _mk_teacher_record("P003", "PD_Receptor", "antagonist",
                               "b_to_a", "family_correct"),
            _mk_teacher_record("P004", "PK_Metabolism", "inhibits_CYP3A4",
                               "a_to_b", "near_miss"),
        ]
        with open(teacher, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        cmd = [
            sys.executable, "-m", "src.teacher.build_preference_pairs",
            "--teacher_clean", str(teacher),
            "--output",        str(out),
            "--report",        str(report),
            "--max_per_pair",  "4",  # allow all strategies to trigger
            "--seed",          "0",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"

        prefs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert prefs, "no preferences emitted"

        # P001: directional, full_correct -> direction_flip + evidence_drop + family_swap expected
        p001 = [p for p in prefs if p["pair_id"] == "P001"]
        types_001 = {p["mirror_type"] for p in p001}
        assert "direction_flip" in types_001, types_001
        assert "evidence_drop" in types_001,  types_001
        assert "family_swap"   in types_001,  types_001

        # P002: bidirectional -> direction_flip must NOT appear
        p002 = [p for p in prefs if p["pair_id"] == "P002"]
        types_002 = {p["mirror_type"] for p in p002}
        assert "direction_flip" not in types_002, (
            f"direction_flip erroneously produced for bidirectional pair: {types_002}")

        # P004: near_miss -> abstain_unsafe expected
        p004 = [p for p in prefs if p["pair_id"] == "P004"]
        types_004 = {p["mirror_type"] for p in p004}
        assert "abstain_unsafe" in types_004, types_004

        # Verify direction_flip actually flipped the tag
        df = next(p for p in p001 if p["mirror_type"] == "direction_flip")
        chosen = json.loads(df["chosen"])
        rejected = json.loads(df["rejected"])
        assert chosen["final_answer"]["direction_tag"] == "a_to_b"
        assert rejected["final_answer"]["direction_tag"] == "b_to_a"

        # Verify evidence_drop cleared evidence_ids
        ed = next(p for p in p001 if p["mirror_type"] == "evidence_drop")
        rejected_ed = json.loads(ed["rejected"])
        for s in rejected_ed["steps"]:
            assert s["evidence_ids"] == [], s

        # Verify family_swap changed family to a confusion candidate
        fs = next(p for p in p001 if p["mirror_type"] == "family_swap")
        rejected_fs = json.loads(fs["rejected"])
        assert rejected_fs["final_answer"]["family"] != "PK_Metabolism"

        # Verify abstain_unsafe has abstain=True on chosen
        au = next(p for p in p004 if p["mirror_type"] == "abstain_unsafe")
        chosen_au = json.loads(au["chosen"])
        assert chosen_au["final_answer"]["abstain"] is True
        assert chosen_au["final_answer"]["family"] == "abstain"

        print(f"PREFERENCE BUILDER OK: {len(prefs)} pairs emitted")
        from collections import Counter
        print(f"  types: {dict(Counter(p['mirror_type'] for p in prefs))}")


if __name__ == "__main__":
    test_preference_builder()
