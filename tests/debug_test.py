
#!/usr/bin/env python3
"""
调试脚本 - 看看到底用了哪个版本
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("调试：检查使用的是规则版本还是 LLM 版本")
print("=" * 60)

# 测试 1：规则版本
print("\n【测试 1】规则版本 (use_llm=False)")
from moodangels.pipeline import MoodAngelsPipeline

pipe_rule = MoodAngelsPipeline(
    project_root / "data" / "syn_train.json",
    use_llm=False
)
print(f"  pipe_rule.use_llm = {pipe_rule.use_llm}")
print(f"  pipe_rule.agents = {pipe_rule.agents}")
print(f"  pipe_rule.llm_client = {pipe_rule.llm_client}")

# 测试 2：LLM 版本
print("\n【测试 2】LLM 版本 (use_llm=True)")
try:
    pipe_llm = MoodAngelsPipeline(
        project_root / "data" / "syn_train.json",
        use_llm=True,
        llm_api_key="sk-7ccd72838c414662870657b1a8a666c8"
    )
    print(f"  pipe_llm.use_llm = {pipe_llm.use_llm}")
    print(f"  pipe_llm.agents = {pipe_llm.agents}")
    print(f"  pipe_llm.llm_client = {pipe_llm.llm_client}")
    
    # 检查 agent 类名
    print(f"\n  Agent 类:")
    for name, cls in pipe_llm.agents.items():
        print(f"    {name}: {cls.__name__}")
        
except Exception as e:
    print(f"  ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("调试完成！")
print("=" * 60)

