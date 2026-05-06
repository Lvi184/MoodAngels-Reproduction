
#!/usr/bin/env python3
"""
调试单个案例，看 LLM 有没有被调用 - 纯 ASCII 版本
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from moodangels.pipeline import MoodAngelsPipeline

# 加载一个案例
test_data_path = project_root / "data" / "syn_test.json"
with open(test_data_path, "r", encoding="utf-8") as f:
    test_cases = json.load(f)
case = test_cases[0]

print("=" * 70)
print(f"Debug case: {case['id']}")
print("=" * 70)

# ========== Rule version ==========
print("\n" + "=" * 30)
print("1. Rule version")
print("=" * 30)

pipe_rule = MoodAngelsPipeline(
    project_root / "data" / "syn_train.json",
    use_llm=False
)
result_rule = pipe_rule.diagnose_dict(case, agent="multi")
print(f"Rule version - label: {result_rule.label}, confidence: {result_rule.confidence}")
print(f"Rule version - agent: {result_rule.agent}")

# ========== LLM version ==========
print("\n" + "=" * 30)
print("2. LLM version")
print("=" * 30)

try:
    pipe_llm = MoodAngelsPipeline(
        project_root / "data" / "syn_train.json",
        use_llm=True,
        llm_api_key="sk-7ccd72838c414662870657b1a8a666c8"
    )
    
    print("OK: Pipeline initialized")
    print(f"   Using agent class: {pipe_llm.agents['multi'].__name__}")
    
    print("\nCalling DeepSeek API...")
    result_llm = pipe_llm.diagnose_dict(case, agent="multi")
    
    print(f"OK: LLM version - label: {result_llm.label}, confidence: {result_llm.confidence}")
    print(f"OK: LLM version - agent: {result_llm.agent}")
    
    print(f"\nCompare:")
    print(f"   Rule version: {result_rule.label}")
    print(f"   LLM version:  {result_llm.label}")
    print(f"   Same: {'YES' if result_rule.label == result_llm.label else 'NO'}")
    
except Exception as e:
    print(f"\nERROR in LLM version: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)

