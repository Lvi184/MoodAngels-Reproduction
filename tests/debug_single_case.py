
#!/usr/bin/env python3
"""
调试单个案例，看 LLM 有没有被调用
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
print(f"调试案例: {case['id']}")
print("=" * 70)

# ========== 规则版本 ==========
print("\n" + "=" * 30)
print("1. 规则版本")
print("=" * 30)

pipe_rule = MoodAngelsPipeline(
    project_root / "data" / "syn_train.json",
    use_llm=False
)
result_rule = pipe_rule.diagnose_dict(case, agent="multi")
print(f"规则版本 - label: {result_rule.label}, confidence: {result_rule.confidence}")
print(f"规则版本 - agent: {result_rule.agent}")

# ========== LLM 版本 ==========
print("\n" + "=" * 30)
print("2. LLM 版本 (如果 API 调用失败会显示)")
print("=" * 30)

try:
    pipe_llm = MoodAngelsPipeline(
        project_root / "data" / "syn_train.json",
        use_llm=True,
        llm_api_key="sk-7ccd72838c414662870657b1a8a666c8"
    )
    
    print("✅ Pipeline 初始化成功")
    print(f"   使用 agent 类: {pipe_llm.agents['multi'].__name__}")
    
    print("\n⏳ 正在调用 DeepSeek API...")
    result_llm = pipe_llm.diagnose_dict(case, agent="multi")
    
    print(f"✅ LLM 版本 - label: {result_llm.label}, confidence: {result_llm.confidence}")
    print(f"✅ LLM 版本 - agent: {result_llm.agent}")
    
    print(f"\n📊 对比:")
    print(f"   规则版本: {result_rule.label}")
    print(f"   LLM 版本:  {result_llm.label}")
    print(f"   是否相同: {'是' if result_rule.label == result_llm.label else '否'}")
    
except Exception as e:
    print(f"\n❌ LLM 版本错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)

