
#!/usr/bin/env python3
"""
三者对比：规则版本 vs LLM+MoodAngels vs 纯 LLM 基线
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from moodangels.pipeline import MoodAngelsPipeline
from moodangels.llm_client import DeepSeekClient
from tests.baseline_llm_only import diagnose_with_llm_only


def calculate_metrics(y_true, y_pred):
    """计算分类指标"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom != 0 else 0
    
    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f10 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0
    
    prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f11 = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) > 0 else 0
    
    macro_f1 = (f10 + f11) / 2
    
    return {
        "acc": acc,
        "mcc": mcc,
        "macro_f1": macro_f1,
        "class_0": {"precision": prec0, "recall": rec0, "f1": f10},
        "class_1": {"precision": prec1, "recall": rec1, "f1": f11},
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }


def main():
    print("=" * 90)
    print("MoodAngels 对比研究：规则版本 vs LLM+MoodAngels vs 纯 LLM 基线")
    print("=" * 90)
    
    # ========== 配置区域 ==========
    DEEPSEEK_API_KEY = "sk-7ccd72838c414662870657b1a8a666c8"
    TEST_LIMIT = None  # 测试前 N 个案例，None 表示全部
    # ===============================
    
    # 加载数据
    test_data_path = project_root / "data" / "syn_test.json"
    print(f"\n[1/6] 加载测试数据: {test_data_path}")
    
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    if TEST_LIMIT:
        test_cases = test_cases[:TEST_LIMIT]
    
    print(f"      共 {len(test_cases)} 个测试案例")
    
    # 初始化所有 pipelines
    print("\n[2/6] 初始化 Pipelines...")
    
    pipe_rule = MoodAngelsPipeline(
        project_root / "data" / "syn_train.json",
        use_llm=False
    )
    
    pipe_llm_moodangels = MoodAngelsPipeline(
        project_root / "data" / "syn_train.json",
        use_llm=True,
        llm_api_key=DEEPSEEK_API_KEY
    )
    
    llm_client = DeepSeekClient(api_key=DEEPSEEK_API_KEY)
    print("      OK: 所有初始化完成")
    
    # 运行诊断
    print("\n[3/6] 开始诊断...")
    
    results = {
        "rule": {"true": [], "pred": [], "details": []},
        "llm_moodangels": {"true": [], "pred": [], "details": []},
        "llm_only": {"true": [], "pred": [], "details": []}
    }
    
    for i, case in enumerate(test_cases, 1):
        case_id = case["id"]
        true_label = case["mood_disorder"]
        
        # 1. 规则版本
        res_rule = pipe_rule.diagnose_dict(case, agent="multi")
        pred_rule = res_rule.label
        
        # 2. LLM + MoodAngels
        res_llm_ma = pipe_llm_moodangels.diagnose_dict(case, agent="multi")
        pred_llm_ma = res_llm_ma.label
        
        # 3. 纯 LLM 基线
        pred_llm_only = diagnose_with_llm_only(llm_client, case)
        
        # 保存结果
        for method, pred in [
            ("rule", pred_rule),
            ("llm_moodangels", pred_llm_ma),
            ("llm_only", pred_llm_only)
        ]:
            results[method]["true"].append(true_label)
            results[method]["pred"].append(pred)
            results[method]["details"].append({
                "id": case_id,
                "true": true_label,
                "pred": pred
            })
        
        # 进度
        if i % 5 == 0 or i == len(test_cases):
            print(f"      进度: {i}/{len(test_cases)} ({100*i/len(test_cases):.1f}%)")
    
    # 计算指标
    print("\n[4/6] 计算各版本指标...")
    metrics = {}
    for method in ["rule", "llm_moodangels", "llm_only"]:
        metrics[method] = calculate_metrics(
            results[method]["true"],
            results[method]["pred"]
        )
    
    # 输出对比表格
    print("\n[5/6] 结果对比汇总")
    print("=" * 90)
    
    print("\n" + "-" * 90)
    print(f"{'指标':<15} | {'规则版本':<15} | {'LLM+MoodAngels':<20} | {'纯 LLM 基线':<20}")
    print("-" * 90)
    print(f"{'ACC (准确率)':<15} | {metrics['rule']['acc']*100:>12.2f}% | {metrics['llm_moodangels']['acc']*100:>17.2f}% | {metrics['llm_only']['acc']*100:>17.2f}%")
    print(f"{'MCC':<15} | {metrics['rule']['mcc']:>15.4f} | {metrics['llm_moodangels']['mcc']:>20.4f} | {metrics['llm_only']['mcc']:>20.4f}")
    print(f"{'Macro F1':<15} | {metrics['rule']['macro_f1']:>15.4f} | {metrics['llm_moodangels']['macro_f1']:>20.4f} | {metrics['llm_only']['macro_f1']:>20.4f}")
    print("-" * 90)
    
    print("\n类别 0 (无情绪障碍) 指标:")
    print(f"{'指标':<15} | {'规则版本':<15} | {'LLM+MoodAngels':<20} | {'纯 LLM 基线':<20}")
    print("-" * 90)
    print(f"{'Precision':<15} | {metrics['rule']['class_0']['precision']:>15.4f} | {metrics['llm_moodangels']['class_0']['precision']:>20.4f} | {metrics['llm_only']['class_0']['precision']:>20.4f}")
    print(f"{'Recall':<15} | {metrics['rule']['class_0']['recall']:>15.4f} | {metrics['llm_moodangels']['class_0']['recall']:>20.4f} | {metrics['llm_only']['class_0']['recall']:>20.4f}")
    print(f"{'F1':<15} | {metrics['rule']['class_0']['f1']:>15.4f} | {metrics['llm_moodangels']['class_0']['f1']:>20.4f} | {metrics['llm_only']['class_0']['f1']:>20.4f}")
    print("-" * 90)
    
    print("\n类别 1 (有情绪障碍) 指标:")
    print(f"{'指标':<15} | {'规则版本':<15} | {'LLM+MoodAngels':<20} | {'纯 LLM 基线':<20}")
    print("-" * 90)
    print(f"{'Precision':<15} | {metrics['rule']['class_1']['precision']:>15.4f} | {metrics['llm_moodangels']['class_1']['precision']:>20.4f} | {metrics['llm_only']['class_1']['precision']:>20.4f}")
    print(f"{'Recall':<15} | {metrics['rule']['class_1']['recall']:>15.4f} | {metrics['llm_moodangels']['class_1']['recall']:>20.4f} | {metrics['llm_only']['class_1']['recall']:>20.4f}")
    print(f"{'F1':<15} | {metrics['rule']['class_1']['f1']:>15.4f} | {metrics['llm_moodangels']['class_1']['f1']:>20.4f} | {metrics['llm_only']['class_1']['f1']:>20.4f}")
    print("-" * 90)
    
    # 找出赢家
    print("\n[6/6] 🏆 各指标优胜者")
    print("=" * 90)
    
    acc_values = [("规则版本", metrics['rule']['acc']), 
                   ("LLM+MoodAngels", metrics['llm_moodangels']['acc']),
                   ("纯 LLM 基线", metrics['llm_only']['acc'])]
    acc_winner = max(acc_values, key=lambda x: x[1])
    
    mcc_values = [("规则版本", metrics['rule']['mcc']), 
                   ("LLM+MoodAngels", metrics['llm_moodangels']['mcc']),
                   ("纯 LLM 基线", metrics['llm_only']['mcc'])]
    mcc_winner = max(mcc_values, key=lambda x: x[1])
    
    f1_values = [("规则版本", metrics['rule']['macro_f1']), 
                  ("LLM+MoodAngels", metrics['llm_moodangels']['macro_f1']),
                  ("纯 LLM 基线", metrics['llm_only']['macro_f1'])]
    f1_winner = max(f1_values, key=lambda x: x[1])
    
    print(f"  ACC 最高:    {acc_winner[0]} ({acc_winner[1]*100:.2f}%)")
    print(f"  MCC 最高:    {mcc_winner[0]} ({mcc_winner[1]:.4f})")
    print(f"  Macro F1 最高: {f1_winner[0]} ({f1_winner[1]:.4f})")
    
    # 保存完整结果
    output_path = project_root / "tests" / "comparison_all_three.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(test_cases),
            "metrics": metrics,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n完整结果已保存到: {output_path}")
    print("\n" + "=" * 90)
    print("论文中的典型结论: MoodAngels 框架应该优于纯 LLM 基线！")
    print("=" * 90)


if __name__ == "__main__":
    try:
        import openai
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("\n请先运行: pip install openai")
        sys.exit(1)
    
    main()

