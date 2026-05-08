from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class DiagnosisResult:
    agent: str
    label: int
    confidence: float
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        label_text = "有情绪障碍" if self.label == 1 else "无情绪障碍"
        return f"{self.agent}: {label_text} (置信度: {self.confidence:.2f})\n理由: {'; '.join(self.reasons)}"

@dataclass
class DebateTurn:
    speaker: str
    text: str

@dataclass
class MultiDiagnosisResult(DiagnosisResult):
    single_agent_results: List[DiagnosisResult] = field(default_factory=list)
    debate: List[DebateTurn] = field(default_factory=list)

@dataclass
class MoodCase:
    id: str
    digit_id: Optional[int]
    raw: Dict[str, Any]
    label: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MoodCase":
        return cls(id=str(d.get("id", d.get("digit_id", "unknown"))), digit_id=d.get("digit_id"), raw=d, label=d.get("mood_disorder"))
