from __future__ import annotations
from typing import Dict, Any, List, Tuple
from .schemas import MoodCase

GROUPS = {
    "depression": ["hama_Q6", "phq9_Q2", "phq9_total", "hamd_Q1", "hamd_total", "bprs_Q9"],
    "interest_energy": ["phq9_Q1", "phq9_Q4", "hamd_Q7", "hamd_Q22", "shaps_total"],
    "suicide": ["phq9_Q9", "hamd_Q3"],
    "anxiety": ["hama_total", "gad7_total"],
    "insomnia": ["hama_Q4", "hamd_Q4", "psqi_total"],
    "mania_bipolar": ["hcl32_total", "mdq_total", "ymrs_total"],
}

# conservative thresholds for common scale/item ranges in MoodSyn-style data
THRESHOLDS = {
    "hama_Q4": 1, "hama_Q6": 1, "phq9_Q1": 1, "phq9_Q2": 1, "phq9_Q4": 1, "phq9_Q9": 1,
    "hamd_Q1": 1, "hamd_Q3": 1, "hamd_Q4": 1, "hamd_Q7": 1, "hamd_Q22": 1, "bprs_Q9": 2,
    "hama_total": 7, "gad7_total": 5, "phq9_total": 5, "hamd_total": 8, "psqi_total": 6,
    "shaps_total": 30, "hcl32_total": 14, "mdq_total": 7, "ymrs_total": 6,
}

def _extract_scores(case: MoodCase) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    raw = case.raw
    for k, v in raw.items():
        if k.endswith("_score") and isinstance(v, (int, float)):
            base = k.replace("_score", "")
            scores[base] = float(v)
    for item in raw.get("mood_disorder_related_performance", []) or []:
        for k, v in item.items():
            if k.endswith("_score") and isinstance(v, (int, float)):
                scores[k.replace("_score", "")] = float(v)
    return scores

def analyze_granular(case: MoodCase) -> Dict[str, Any]:
    scores = _extract_scores(case)
    group_results = {}
    weighted_total = 0.0
    weight_sum = 0.0
    for group, keys in GROUPS.items():
        hits: List[Tuple[str, float, float]] = []
        vals = []
        for key in keys:
            if key in scores and scores[key] >= 0:
                thr = THRESHOLDS.get(key, 1)
                sev = min(scores[key] / max(thr, 1), 3.0) / 3.0
                vals.append(sev)
                if scores[key] >= thr:
                    hits.append((key, scores[key], thr))
        score = sum(vals) / len(vals) if vals else 0.0
        group_results[group] = {"score": round(score, 3), "hits": hits, "n_observed": len(vals)}
        w = 1.2 if group in {"depression", "suicide", "mania_bipolar"} else 1.0
        weighted_total += w * score
        weight_sum += w
    overall = weighted_total / weight_sum if weight_sum else 0.0
    contradiction = detect_contradictions(group_results)
    return {"scores": scores, "groups": group_results, "overall": round(overall, 3), "contradictions": contradiction}

def detect_contradictions(groups: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    dep = groups.get("depression", {}).get("score", 0)
    anxiety = groups.get("anxiety", {}).get("score", 0)
    mania = groups.get("mania_bipolar", {}).get("score", 0)
    suicide = groups.get("suicide", {}).get("score", 0)
    if dep < 0.25 and suicide >= 0.33:
        notes.append("suicidal markers are present despite weak depression markers; flag for clinical review")
    if mania >= 0.45 and dep >= 0.35:
        notes.append("both manic/bipolar and depressive indicators are elevated; consider bipolar differential")
    if anxiety >= 0.6 and dep < 0.25:
        notes.append("anxiety is high while core depressive markers are weak; avoid mood-disorder overcall")
    return notes
