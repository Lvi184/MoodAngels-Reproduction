from __future__ import annotations
import json, csv
from pathlib import Path
from typing import List, Dict, Any
from .schemas import MoodCase

TOTAL_SCORE_FIELDS = [
    "hama_total_score", "gad7_total_score", "phq9_total_score", "hamd_total_score",
    "bprs_total_score", "psqi_total_score", "shaps_total_score", "hcl32_total_score",
    "das_total_score", "ssrs_total_score", "mdq_total_score", "ymrs_total_score",
]

SELECTED_ITEM_KEYS = [
    "hama_Q4", "hama_Q6", "hama_total", "gad7_total", "phq9_Q1", "phq9_Q2",
    "phq9_Q4", "phq9_Q9", "phq9_total", "hamd_Q1", "hamd_Q3", "hamd_Q4",
    "hamd_Q7", "hamd_Q22", "hamd_total", "bprs_Q9",
]

def load_cases(path: str | Path) -> List[MoodCase]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return [MoodCase.from_dict(x) for x in data]

def flatten_case(case: MoodCase) -> Dict[str, Any]:
    out = {k: case.raw.get(k) for k in TOTAL_SCORE_FIELDS}
    for item in case.raw.get("mood_disorder_related_performance", []) or []:
        for k, v in item.items():
            if k.endswith("_score") or k.endswith("_description") or k == "correlation_to_mood_disorder":
                out[k] = v
    return out

def case_to_text(case: MoodCase, include_label: bool=False) -> str:
    parts = []
    for k, v in case.raw.items():
        if k.endswith("_description") and isinstance(v, str):
            parts.append(v)
    for item in case.raw.get("mood_disorder_related_performance", []) or []:
        for k, v in item.items():
            if k.endswith("_description") and isinstance(v, str):
                parts.append(v)
    if include_label and case.label is not None:
        parts.append(f"诊断标签 mood_disorder={case.label}")
    return "\n".join(parts)
