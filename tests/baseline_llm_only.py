
#!/usr/bin/env python3
"""
纯 LLM 基线测试 - 直接把案例丢给大模型，不使用 MoodAngels 框架
用于对比研究
"""

import sys
import os
import json
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from moodangels.llm_client import DeepSeekClient


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


def build_case_prompt(case: dict) -> str:
    """构建案例提示词"""
    prompt = f"请诊断以下患者是否患有心境障碍（mood disorder）。\n\n"
    prompt += f"患者信息:\n"
    prompt += f"  ID: {case.get('id', 'N/A')}\n"
    prompt += f"  总体评分:\n"
    prompt += f"    HAMA: {case.get('hama_total_score', 'N/A')} - {case.get('hama_total_score_description', '')}\n"
    prompt += f"    GAD-7: {case.get('gad7_total_score', 'N/A')} - {case.get('gad7_total_score_description', '')}\n"
    prompt += f"    PHQ-9: {case.get('phq9_total_score', 'N/A')} - {case.get('phq9_total_score_description', '')}\n"
    prompt += f"    HAMD: {case.get('hamd_total_score', 'N/A')} - {case.get('hamd_total_score_description', '')}\n"
    prompt += f"    BPRS: {case.get('bprs_total_score', 'N/A')} - {case.get('bprs_total_score_description', '')}\n"
    prompt += f"    PSQI: {case.get('psqi_total_score', 'N/A')} - {case.get('psqi_total_score_description', '')}\n"
    prompt += f"    SHAPS: {case.get('shaps_total_score', 'N/A')} - {case.get('shaps_total_score_description', '')}\n"
    prompt += f"    HCL-32: {case.get('hcl32_total_score', 'N/A')} - {case.get('hcl32_total_score_description', '')}\n"
    prompt += f"    DAS: {case.get('das_total_score', 'N/A')} - {case.get('das_total_score_description', '')}\n"
    prompt += f"    MDQ: {case.get('mdq_total_score', 'N/A')} - {case.get('mdq_total_score_description', '')}\n"
    prompt += f"    YMRS: {case.get('ymrs_total_score', 'N/A')} - {case.get('ymrs_total_score_description', '')}\n"
    
    return prompt


def diagnose_with_llm_only(llm: DeepSeekClient, case: dict) -> int:
    """纯 LLM 诊断，不使用 MoodAngels 框架"""
    system_prompt = """你是一名专业的精神科诊断专家。请根据提供的患者信息，判断其是否患有心境障碍（mood disorder）。

心境障碍包含：
- 抑郁症
- 双相情感障碍（抑郁或躁狂发作）

请严格按照以下 JSON 格式输出：
{
  "label": 0 或 1（0 表示无心境障碍，1 表示有心境障碍）,
  "confidence": 0.0-1.0 之间的置信度,
  "reasoning": "简短的诊断理由"
}"""
    
    user_prompt = build_case_prompt(case)
    user_prompt += "\n请给出诊断结果。"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    result = llm.chat_with_json(messages, temperature=0.3, max_tokens=1000)
    
    label = result.get("label", 0)
    if label not in [0, 1]:
        label = 0
    
    return label


def main():
    print("=" * 80)
    print("MoodAngels 对比研究：纯 LLM 基线版本")
    print("=" * 80)
    
    # 配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # 从环境变量加载 API Key
    TEST_LIMIT = None  # 测试前 N 个案例，None 表示全部
    
    if not DEEPSEEK_API_KEY:
        print("错误：需要设置 DEEPSEEK_API_KEY 环境变量！")
        print("设置方式：")
        print("  Windows: set DEEPSEEK_API_KEY=your-api-key")
        print("  Linux/Mac: export DEEPSEEK_API_KEY=your-api-key")
        sys.exit(1)
    
    # 加载数据
    test_data_path = project_root / "data" / "syn_test.json"
    print(f"\n[1/4] 加载测试数据: {test_data_path}")
    
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    if TEST_LIMIT:
        test_cases = test_cases[:TEST_LIMIT]
    
    print(f"      共 {len(test_cases)} 个测试案例")
    
    # 初始化 LLM 客户端
    print("\n[2/4] 初始化 DeepSeek 客户端...")
    llm = DeepSeekClient(api_key=DEEPSEEK_API_KEY)
    print("      OK")
    
    # 运行诊断
    print("\n[3/4] 开始诊断（纯 LLM 版本）...")
    y_true = []
    y_pred = []
    predictions = []
    errors = []
    
    for i, case in enumerate(test_cases, 1):
        case_id = case.get("id", f"case_{i}")
        true_label = case["mood_disorder"]
        
        try:
            pred_label = diagnose_with_llm_only(llm, case)
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            predictions.append({
                "id": case_id,
                "true": true_label,
                "pred": pred_label
            })
            
            if i % 5 == 0 or i == len(test_cases):
                current_metrics = calculate_metrics(y_true, y_pred)
                print(f"      进度: {i}/{len(test_cases)} ({100*i/len(test_cases):.1f}%)")
                print(f"             ACC: {current_metrics['acc']*100:.1f}%")
                
        except Exception as e:
            print(f"      案例 {case_id} 出错: {e}")
            errors.append({
                "id": case_id,
                "error": str(e)
            })
    
    # 计算最终指标
    print("\n[4/4] 计算最终指标...")
    metrics = calculate_metrics(y_true, y_pred)
    cm = metrics["confusion_matrix"]
    
    # 输出结果
    print("\n" + "=" * 80)
    print("纯 LLM 基线版本结果")
    print("=" * 80)
    print(f"总案例数: {len(y_true)}")
    print(f"正确数: {cm['tp'] + cm['tn']}")
    print(f"错误数: {cm['fp'] + cm['fn']}")
    print("-" * 80)
    print("\n核心指标:")
    print(f"  ACC (准确率):     {metrics['acc']*100:.2f}%")
    print(f"  MCC (马修斯系数): {metrics['mcc']:.4f}")
    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    
    print("\n类别详细指标:")
    print(f"  类别 0 (无情绪障碍):")
    print(f"    Precision: {metrics['class_0']['precision']:.4f}")
    print(f"    Recall:    {metrics['class_0']['recall']:.4f}")
    print(f"    F1:        {metrics['class_0']['f1']:.4f}")
    print(f"  类别 1 (有情绪障碍):")
    print(f"    Precision: {metrics['class_1']['precision']:.4f}")
    print(f"    Recall:    {metrics['class_1']['recall']:.4f}")
    print(f"    F1:        {metrics['class_1']['f1']:.4f}")
    
    print("\n混淆矩阵:")
    print(f"                  预测为 0    预测为 1")
    print(f"  真实为 0        {cm['tn']:4d}        {cm['fp']:4d}")
    print(f"  真实为 1        {cm['fn']:4d}        {cm['tp']:4d}")
    
    wrong_cases = [p for p in predictions if p["true"] != p["pred"]]
    if wrong_cases:
        print(f"\n错误案例 ({len(wrong_cases)} 个):")
        for wc in wrong_cases[:10]:
            print(f"  {wc['id']}: 真实={wc['true']}, 预测={wc['pred']}")
        if len(wrong_cases) > 10:
            print(f"  ... 还有 {len(wrong_cases) - 10} 个")
    
    output_path = project_root / "tests" / "test_results_baseline_llm_only.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": "baseline_llm_only",
            "total": len(y_true),
            "metrics": metrics,
            "predictions": predictions,
            "errors": errors
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        import openai
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("\n请先运行: pip install openai")
        sys.exit(1)
    
    main()

