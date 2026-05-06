from __future__ import annotations
from typing import List
from .schemas import MoodCase, DiagnosisResult, MultiDiagnosisResult, DebateTurn
from .granular import analyze_granular
from .retrieval import RetrievalStore
from .data import case_to_text

class BaseAngel:
    name = "BaseAngel"
    def __init__(self, store: RetrievalStore | None = None):
        self.store = store or RetrievalStore()

    def _rule_decision(self, analysis: dict, retrieval_bias: float = 0.0) -> tuple[int, float, List[str]]:
        g = analysis["groups"]
        dep = g["depression"]["score"]; interest = g["interest_energy"]["score"]
        suicide = g["suicide"]["score"]; mania = g["mania_bipolar"]["score"]
        overall = analysis["overall"]
        score = 0.42*dep + 0.18*interest + 0.15*suicide + 0.18*mania + 0.07*overall + retrieval_bias
        label = 1 if score >= 0.34 or (suicide >= 0.55 and dep >= 0.25) or mania >= 0.62 else 0
        confidence = min(0.98, max(0.52, abs(score-0.34)*1.7 + 0.55))
        reasons = [f"granular score={overall:.2f}; depression={dep:.2f}, interest/energy={interest:.2f}, suicide={suicide:.2f}, mania/bipolar={mania:.2f}"]
        if retrieval_bias:
            reasons.append(f"case-retrieval adjustment={retrieval_bias:+.2f}")
        reasons.extend(analysis.get("contradictions", []))
        return label, round(confidence,3), reasons

class AngelR(BaseAngel):
    name = "Angel.R"
    def diagnose(self, case: MoodCase) -> DiagnosisResult:
        analysis = analyze_granular(case)
        matches = self.store.symptom_matches(case_to_text(case), k=5)
        label, conf, reasons = self._rule_decision(analysis)
        return DiagnosisResult(self.name, label, conf, reasons, {"granular": analysis, "symptom_matches": matches})

class AngelD(BaseAngel):
    name = "Angel.D"
    def diagnose(self, case: MoodCase) -> DiagnosisResult:
        analysis = analyze_granular(case)
        sims = self.store.similar_cases(case, k=5)
        pos = sum(1 for s in sims if s.get("label") == 1); neg = sum(1 for s in sims if s.get("label") == 0)
        bias = 0.04 * (pos - neg) / max(len(sims), 1)
        label, conf, reasons = self._rule_decision(analysis, bias)
        return DiagnosisResult(self.name, label, conf, reasons, {"granular": analysis, "similar_cases": sims})

class AngelC(BaseAngel):
    name = "Angel.C"
    def diagnose(self, case: MoodCase) -> DiagnosisResult:
        analysis = analyze_granular(case)
        sims = self.store.similar_cases(case, k=5)
        weighted = sum((s.get("score",0.0)) * (1 if s.get("label")==1 else -1) for s in sims)
        denom = sum(abs(s.get("score",0.0)) for s in sims) or 1.0
        bias = 0.06 * weighted / denom
        label, conf, reasons = self._rule_decision(analysis, bias)
        comparison = [f"{s['id']}: label={s.get('label')} similarity={s.get('score')}" for s in sims]
        reasons.append("comparative retrieval: " + "; ".join(comparison[:5]) if comparison else "no comparable cases available")
        return DiagnosisResult(self.name, label, conf, reasons, {"granular": analysis, "comparative_cases": sims})

class MultiAngels(BaseAngel):
    name = "multi-Angels"
    def diagnose(self, case: MoodCase) -> MultiDiagnosisResult:
        agents = [AngelR(self.store), AngelD(self.store), AngelC(self.store)]
        results = [a.diagnose(case) for a in agents]
        votes = sum(r.label for r in results)
        debate: List[DebateTurn] = []
        if votes in {0,3}:
            label = 1 if votes == 3 else 0
            debate.append(DebateTurn("Judge", "All single agents agree; no debate needed."))
        else:
            pos = [r for r in results if r.label == 1]
            neg = [r for r in results if r.label == 0]
            debate.append(DebateTurn("Positive", "Mood-disorder evidence: " + " | ".join(x.reasons[0] for x in pos)))
            debate.append(DebateTurn("Negative", "Against mood disorder: " + " | ".join(x.reasons[0] for x in neg)))
            # weighted judge: confidence-weighted majority
            score = sum((1 if r.label else -1) * r.confidence for r in results)
            label = 1 if score >= 0 else 0
            debate.append(DebateTurn("Judge", f"Confidence-weighted vote={score:.3f}; final label={label}."))
        avg_conf = sum(r.confidence for r in results if r.label == label) / max(1, sum(1 for r in results if r.label == label))
        reasons = [f"votes={votes}/3 for mood disorder", *[f"{r.agent}: label={r.label}, confidence={r.confidence}" for r in results]]
        return MultiDiagnosisResult(self.name, label, round(avg_conf,3), reasons, {"case_id": case.id}, results, debate)
