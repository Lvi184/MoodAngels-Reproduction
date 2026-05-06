
#!/usr/bin/env python3
"""
直接对比规则版本 vs LLM 版本 - 前 5 个案例
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from moodangels.pipeline import MoodAngelsPipeline

# 加载前 5 个案例
test_data_path = project_root / "data" / "syn_test.json"
with open(test_data_path, "r", encoding="utf-8") as f:
    test_cases = json.load(f)[:5]

print("=" * 80)
print("MoodAngels: 规则版本 vs LLM 版本对比 (前 5 个案例)")
print("=" * 80)

# 初始化两个 pipeline
print("\n[1/3] 初始化 Pipelines...")
pipe_rule = MoodAngelsPipeline(
    project_root / "data" / "syn_train.json",
    use_llm=False
)
pipe_llm = MoodAngelsPipeline(
    project_root / "data" / "syn_train.json",
    use_llm=True,
    llm_api_key="sk-7ccd72838c414662870657b1a8a666c8"
)
print("OK: 两个 Pipeline 都初始化好了")

# 运行对比
print("\n[2/3] 开始诊断...")
results = []
for i, case in enumerate(test_cases, 1):
    case_id = case["id"]
    true_label = case["mood_disorder"]
    
    # 规则版本
    res_rule = pipe_rule.diagnose_dict(case, agent="multi")
    pred_rule = res_rule.label
    
    # LLM 版本
    res_llm = pipe_llm.diagnose_dict(case, agent="multi")
    pred_llm = res_llm.label
    
    results.append({
        "id": case_id,
        "true": true_label,
        "rule": pred_rule,
        "llm": pred_llm
    })
    
    print(f"\n案例 {i} ({case_id}):")
    print(f"  真实标签:      {true_label}")
    print(f"  规则版本预测:  {pred_rule}")
    print(f"  LLM 版本预测:   {pred_llm}")
    print(f"  规则版本正确:  {'YES' if pred_rule == true_label else 'NO'}")
    print(f"  LLM 版本正确:   {'YES' if pred_llm == true_label else 'NO'}")
    print(f"  两版本一致:     {'YES' if pred_rule == pred_llm else 'NO'}")

# 总结
print("\n" + "=" * 80)
print("[3/3] 总结")
print("=" * 80)

rule_correct = sum(1 for r in results if r["rule"] == r["true"])
llm_correct = sum(1 for r in results if r["llm"] == r["true"])
same_count = sum(1 for r in results if r["rule"] == r["llm"])

print(f"\n总案例数: {len(results)}")
print(f"\n规则版本:")
print(f"  正确数: {rule_correct}/{len(results)}")
print(f"  准确率: {rule_correct/len(results)*100:.1f}%")
print(f"\nLLM 版本:")
print(f"  正确数: {llm_correct}/{len(results)}")
print(f"  准确率: {llm_correct/len(results)*100:.1f}%")
print(f"\n两版本预测一致: {same_count}/{len(results)}")
print(f"两版本预测不一致: {len(results)-same_count}/{len(results)}")

print("\n" + "=" * 80)
print("提示: 如果两版本完全一致，说明 LLM 版本没有正确运行！")
print("=" * 80)

