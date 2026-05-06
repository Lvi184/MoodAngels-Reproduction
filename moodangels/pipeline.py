
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
from .data import load_cases
from .schemas import MoodCase
from .retrieval import RetrievalStore
from .agents import AngelR, AngelD, AngelC, MultiAngels
from .agents_llm import AngelR_LLM, AngelD_LLM, AngelC_LLM, MultiAngels_LLM
from .llm_client import DeepSeekClient

# 规则版本
AGENTS_RULE = {"raw": AngelR, "display": AngelD, "compare": AngelC, "multi": MultiAngels}
# LLM 版本
AGENTS_LLM = {"raw": AngelR_LLM, "display": AngelD_LLM, "compare": AngelC_LLM, "multi": MultiAngels_LLM}


class MoodAngelsPipeline:
    def __init__(
        self, 
        retrieval_data: Optional[str | Path] = None,
        use_llm: bool = False,
        llm_api_key: Optional[str] = None
    ):
        """
        初始化 MoodAngels Pipeline
        
        Args:
            retrieval_data: 检索数据路径
            use_llm: 是否使用 LLM 版本（True=使用 DeepSeek API，False=使用规则版本）
            llm_api_key: DeepSeek API Key（如果 use_llm=True 则需要）
        """
        cases = load_cases(retrieval_data) if retrieval_data else []
        self.store = RetrievalStore(cases)
        self.use_llm = use_llm
        
        if use_llm:
            self.llm_client = DeepSeekClient(api_key=llm_api_key)
            self.agents = AGENTS_LLM
        else:
            self.llm_client = None
            self.agents = AGENTS_RULE

    def diagnose_dict(self, case_dict: Dict[str, Any], agent: str = "multi"):
        case = MoodCase.from_dict(case_dict)
        cls = self.agents.get(agent)
        if not cls:
            raise ValueError(f"Unknown agent {agent}; choose one of {list(self.agents)}")
        
        if self.use_llm:
            return cls(self.store, self.llm_client).diagnose(case)
        else:
            return cls(self.store).diagnose(case)

