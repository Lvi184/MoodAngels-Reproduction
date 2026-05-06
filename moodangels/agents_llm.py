
#!/usr/bin/env python3
"""
基于真实 LLM API 的 MoodAngels 智能体实现
参考官方代码架构
"""

from __future__ import annotations
import json
from typing import List, Dict, Any
from .schemas import MoodCase, DiagnosisResult, MultiDiagnosisResult, DebateTurn
from .granular import analyze_granular
from .retrieval import RetrievalStore
from .data import case_to_text
from .llm_client import DeepSeekClient


def build_case_context(case: MoodCase) -> str:
    """构建案例上下文文本"""
    context = f"案例 ID: {case.id}\n"
    context += f"患者信息: {json.dumps(case.raw, ensure_ascii=False, indent=2)}\n"
    return context


def build_similar_cases_context(store: RetrievalStore, case: MoodCase, k: int = 5) -> str:
    """构建相似案例上下文"""
    sims = store.similar_cases(case, k=k)
    if not sims:
        return "无相似案例"
    
    context = f"\n参考的 {len(sims)} 个相似案例:\n"
    for i, s in enumerate(sims, 1):
        context += f"\n案例 {i}:\n"
        context += f"  ID: {s.get('id', 'N/A')}\n"
        context += f"  标签: {'有情绪障碍' if s.get('label') == 1 else '无情绪障碍'}\n"
        context += f"  相似度: {s.get('score', 0):.3f}\n"
    return context


def build_symptom_matches_context(store: RetrievalStore, case: MoodCase, k: int = 5) -> str:
    """构建症状匹配上下文"""
    matches = store.symptom_matches(case_to_text(case), k=k)
    if not matches:
        return "无匹配症状"
    
    context = f"\n匹配的症状 ({len(matches)} 个):\n"
    for i, m in enumerate(matches, 1):
        context += f"  {i}. {m}\n"
    return context


class BaseLLMAngel:
    """基于 LLM 的基础智能体"""
    
    name = "BaseLLMAngel"
    
    def __init__(self, store: RetrievalStore | None = None, llm_client: DeepSeekClient | None = None):
        self.store = store or RetrievalStore()
        self.llm = llm_client
    
    def _call_llm_for_diagnosis(self, case: MoodCase, extra_context: str = "") -> tuple[int, float, List[str]]:
        """
        调用 LLM 进行诊断
        
        Returns:
            (label, confidence, reasons)
        """
        system_prompt = """你是一名专业的精神科诊断专家。请根据提供的患者信息，判断其是否患有心境障碍（mood disorder）。

心境障碍包含：
- 抑郁症
- 双相情感障碍（抑郁或躁狂发作）

请严格按照以下 JSON 格式输出：
{
  "label": 0 或 1（0 表示无心境障碍，1 表示有心境障碍）,
  "confidence": 0.0-1.0 之间的置信度,
  "reasons": ["原因1", "原因2", "原因3"]
}"""
        
        user_prompt = f"请诊断以下患者是否患有心境障碍：\n\n"
        user_prompt += build_case_context(case)
        
        if extra_context:
            user_prompt += f"\n{extra_context}\n"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 调用 LLM
        result = self.llm.chat_with_json(messages, temperature=0.3, max_tokens=1500)
        
        # 解析结果
        label = result.get("label", 0)
        confidence = result.get("confidence", 0.7)
        reasons = result.get("reasons", ["基于规则判断"])
        
        # 确保 label 是 0 或 1
        if label not in [0, 1]:
            label = 0
        
        return label, confidence, reasons


class AngelR_LLM(BaseLLMAngel):
    """Angel.R: 原始智能体，分析症状"""
    
    name = "Angel.R (LLM)"
    
    def diagnose(self, case: MoodCase) -> DiagnosisResult:
        # 分析粒度
        analysis = analyze_granular(case)
        
        # 获取症状匹配
        symptom_context = build_symptom_matches_context(self.store, case, k=5)
        
        # 调用 LLM
        label, conf, reasons = self._call_llm_for_diagnosis(case, symptom_context)
        
        return DiagnosisResult(self.name, label, conf, reasons, {
            "granular": analysis,
            "symptom_matches": symptom_context
        })


class AngelD_LLM(BaseLLMAngel):
    """Angel.D: 显示智能体，参考相似案例"""
    
    name = "Angel.D (LLM)"
    
    def diagnose(self, case: MoodCase) -> DiagnosisResult:
        # 分析粒度
        analysis = analyze_granular(case)
        
        # 获取相似案例
        similar_context = build_similar_cases_context(self.store, case, k=5)
        
        # 调用 LLM
        label, conf, reasons = self._call_llm_for_diagnosis(case, similar_context)
        
        return DiagnosisResult(self.name, label, conf, reasons, {
            "granular": analysis,
            "similar_cases": similar_context
        })


class AngelC_LLM(BaseLLMAngel):
    """Angel.C: 比较智能体，比较相似案例"""
    
    name = "Angel.C (LLM)"
    
    def diagnose(self, case: MoodCase) -> DiagnosisResult:
        # 分析粒度
        analysis = analyze_granular(case)
        
        # 获取相似案例
        similar_context = build_similar_cases_context(self.store, case, k=5)
        
        extra_context = f"{similar_context}\n请综合比较这些相似案例，给出诊断。"
        
        # 调用 LLM
        label, conf, reasons = self._call_llm_for_diagnosis(case, extra_context)
        
        return DiagnosisResult(self.name, label, conf, reasons, {
            "granular": analysis,
            "comparative_cases": similar_context
        })


class MultiAngels_LLM(BaseLLMAngel):
    """多智能体辩论 + LLM"""
    
    name = "multi-Angels (LLM)"
    
    def diagnose(self, case: MoodCase) -> MultiDiagnosisResult:
        # 初始化三个智能体
        agents = [
            AngelR_LLM(self.store, self.llm),
            AngelD_LLM(self.store, self.llm),
            AngelC_LLM(self.store, self.llm)
        ]
        
        # 分别诊断
        results = [a.diagnose(case) for a in agents]
        
        # 投票
        votes = sum(r.label for r in results)
        debate: List[DebateTurn] = []
        
        if votes in {0, 3}:
            # 全票通过
            label = 1 if votes == 3 else 0
            debate.append(DebateTurn("Judge", "All single agents agree; no debate needed."))
        else:
            # 有分歧，需要辩论
            pos = [r for r in results if r.label == 1]
            neg = [r for r in results if r.label == 0]
            
            debate.append(DebateTurn(
                "Positive",
                "Mood-disorder evidence: " + " | ".join(x.reasons[0] for x in pos)
            ))
            debate.append(DebateTurn(
                "Negative",
                "Against mood disorder: " + " | ".join(x.reasons[0] for x in neg)
            ))
            
            # 置信度加权投票
            score = sum((1 if r.label else -1) * r.confidence for r in results)
            label = 1 if score >= 0 else 0
            
            debate.append(DebateTurn(
                "Judge",
                f"Confidence-weighted vote={score:.3f}; final label={label}."
            ))
        
        # 计算平均置信度
        avg_conf = sum(
            r.confidence for r in results if r.label == label
        ) / max(1, sum(1 for r in results if r.label == label))
        
        reasons = [f"votes={votes}/3 for mood disorder"]
        reasons.extend([f"{r.agent}: label={r.label}, confidence={r.confidence}" for r in results])
        
        return MultiDiagnosisResult(
            self.name, label, round(avg_conf, 3), reasons,
            {"case_id": case.id}, results, debate
        )

