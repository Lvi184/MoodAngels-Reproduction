
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from .data import load_cases, case_to_text
from .schemas import MoodCase, DiagnosisResult
from .retrieval import RetrievalStore
from .agents_llm import AngelR_LLM, AngelD_LLM, AngelC_LLM
from .llm_client import DeepSeekClient
from .debate import debate
from .judge import judge


def parse_judge_result(result_text: str) -> Tuple[int, float, str]:
    """
    解析裁判的输出结果
    返回: (label, confidence, reasoning)
    """
    # 尝试解析 JSON
    try:
        start_idx = result_text.find("{")
        end_idx = result_text.rfind("}") + 1
        if start_idx != -1 and end_idx != -1:
            json_str = result_text[start_idx:end_idx]
            result = json.loads(json_str)
            label = int(result.get("label", 0))
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")
            return label, confidence, reasoning
    except Exception:
        pass
    
    # 回退到启发式解析
    result_text_lower = result_text.lower()
    if any(keyword in result_text_lower for keyword in ["yes", "has", "have", "存在", "有", "1"]):
        return 1, 0.6, result_text[:200]
    elif any(keyword in result_text_lower for keyword in ["no", "not", "没有", "不存在", "0"]):
        return 0, 0.6, result_text[:200]
    
    # 默认返回
    return 0, 0.5, result_text[:200]


def diagnose_with_debate(case: MoodCase, llm_client: DeepSeekClient, retriever: RetrievalStore):
    """
    完整的 MoodAngels 论文复现版本：
    1. 三个 LLM 天使独立诊断
    2. 辩论阶段
    3. 裁判决定
    """
    # 1. 初始化三个天使
    angel_r = AngelR_LLM(retriever, llm_client)
    angel_d = AngelD_LLM(retriever, llm_client)
    angel_c = AngelC_LLM(retriever, llm_client)
    
    # 2. 三个天使独立诊断
    result_r = angel_r.diagnose(case)
    result_d = angel_d.diagnose(case)
    result_c = angel_c.diagnose(case)
    
    # 3. 包装 LLM 调用函数
    def llm_call(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return llm_client.chat(messages, temperature=0.3, max_tokens=1000)
    
    # 4. 辩论阶段
    debate_result = debate(
        result_r,
        result_d,
        result_c,
        llm_call
    )
    
    # 5. 裁判决定
    final_result_text = judge(
        result_r,
        result_d,
        result_c,
        debate_result,
        llm_call
    )
    
    # 6. 解析裁判结果
    label, confidence, reasoning = parse_judge_result(final_result_text)
    
    # 7. 返回结构化结果
    final_result = DiagnosisResult(
        agent="Judge (辩论+裁判)",
        label=label,
        confidence=confidence,
        reasons=[reasoning],
        evidence={"debate": debate_result}
    )
    
    return final_result, {
        "angel_r": result_r,
        "angel_d": result_d,
        "angel_c": result_c,
        "debate": debate_result,
        "judge_raw": final_result_text
    }


class MoodAngelsPipelineFull:
    """
    完整的 MoodAngels 论文复现 Pipeline
    """
    def __init__(
        self, 
        retrieval_data: Optional[str | Path] = None,
        llm_api_key: Optional[str] = None
    ):
        cases = load_cases(retrieval_data) if retrieval_data else []
        self.store = RetrievalStore(cases)
        self.llm_client = DeepSeekClient(api_key=llm_api_key) if llm_api_key else None

    def diagnose_dict(self, case_dict: Dict[str, Any]):
        case = MoodCase.from_dict(case_dict)
        return diagnose_with_debate(case, self.llm_client, self.store)
