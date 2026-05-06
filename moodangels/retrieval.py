from __future__ import annotations
import math, re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from .schemas import MoodCase
from .data import case_to_text

DSM5_MOOD_CRITERIA = [
    {"id":"MDD_A", "category":"Mood Disorder", "text":"Major depressive episode: depressed mood or loss of interest with symptoms such as sleep disturbance, fatigue, guilt, concentration difficulty, psychomotor change, appetite change, or suicidal ideation."},
    {"id":"BIPOLAR_MANIA", "category":"Mood Disorder", "text":"Manic episode: abnormally elevated or irritable mood and increased energy with decreased need for sleep, grandiosity, pressured speech, racing thoughts, distractibility, increased activity, or risky behavior."},
    {"id":"BIPOLAR_HYPOMANIA", "category":"Mood Disorder", "text":"Hypomanic episode: elevated or irritable mood and increased activity lasting several days, observable by others, but not causing marked impairment or psychosis."},
    {"id":"GAD", "category":"Anxiety Disorder", "text":"Generalized anxiety: excessive anxiety and worry with restlessness, fatigue, concentration difficulty, irritability, muscle tension, and sleep disturbance."},
    {"id":"PSYCHOSIS", "category":"Psychotic Disorder", "text":"Psychosis indicators: delusions, hallucinations, disorganized speech, grossly disorganized behavior, negative symptoms, self-talk, suspiciousness."},
    {"id":"NORMAL_LOW", "category":"No current mood disorder", "text":"Low symptom burden: no depressive mood, no anhedonia, no suicidal ideation, intact energy and sleep, and no clinician-rated mood symptoms."},
]

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]

class SimpleVectorIndex:
    def __init__(self, docs: List[str], payloads: Optional[List[dict]]=None):
        self.docs = docs
        self.payloads = payloads or [{} for _ in docs]
        self.df = defaultdict(int)
        self.tfs = []
        for doc in docs:
            c = Counter(tokenize(doc))
            self.tfs.append(c)
            for t in c:
                self.df[t] += 1
        self.n = max(len(docs), 1)
        self.vecs = [self._tfidf(c) for c in self.tfs]

    def _tfidf(self, c: Counter) -> Dict[str, float]:
        vec = {}
        for t, tf in c.items():
            idf = math.log((1 + self.n) / (1 + self.df[t])) + 1
            vec[t] = (1 + math.log(tf)) * idf
        return vec

    @staticmethod
    def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b: return 0.0
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = math.sqrt(sum(v*v for v in a.values())); nb = math.sqrt(sum(v*v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query: str, k: int=5) -> List[Tuple[float, str, dict]]:
        q = self._tfidf(Counter(tokenize(query)))
        scored = [(self._cos(q, v), self.docs[i], self.payloads[i]) for i, v in enumerate(self.vecs)]
        return sorted(scored, key=lambda x: x[0], reverse=True)[:k]

class RetrievalStore:
    def __init__(self, cases: Optional[List[MoodCase]]=None):
        self.cases = cases or []
        self.case_index = SimpleVectorIndex([case_to_text(c, include_label=True) for c in self.cases], [c.raw for c in self.cases]) if self.cases else None
        self.criteria_index = SimpleVectorIndex([d["text"] for d in DSM5_MOOD_CRITERIA], DSM5_MOOD_CRITERIA)

    def similar_cases(self, case: MoodCase, k: int=5) -> List[dict]:
        if not self.case_index:
            return []
        query = case_to_text(case)
        out = []
        for score, _doc, payload in self.case_index.search(query, k=k+1):
            if str(payload.get("id")) == case.id:
                continue
            out.append({"score": round(score, 4), "id": payload.get("id"), "digit_id": payload.get("digit_id"), "label": payload.get("mood_disorder"), "summary": _doc[:800]})
            if len(out) >= k: break
        return out

    def symptom_matches(self, text: str, k: int=5) -> List[dict]:
        return [{"score": round(s,4), **p} for s, _d, p in self.criteria_index.search(text, k=k)]
