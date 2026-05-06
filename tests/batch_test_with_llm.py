
#!/usr/bin/env python3
"""
批量测试脚本 - 支持规则版本和 LLM 版本切换
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from moodangels.pipeline import MoodAngelsPipeline
from moodangels.data import load_cases


def calculate_metrics(y_true, y_pred):
    """计算分类指标: ACC, MCC, Macro F1, Precision, Recall"""
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
    print("=" * 80)
    print("MoodAngels 批量测试 - 支持规则/LLM 切换")
    print("=" * 80)
    
    # ========== 配置区域 ==========
    USE_LLM = True  # 改为 True 使用 DeepSeek API
    DEEPSEEK_API_KEY = "sk-7ccd72838c414662870657b1a8a666c8"  # 你的 API Key
    TEST_LIMIT = None  # 测试前 N 个案例，None 表示全部
    # =============================
    
    if USE_LLM:
        print(f"\n🚀 使用 LLM 版本 (DeepSeek API)")
    else:
        print(f"\n📏 使用规则版本 (启发式公式)")
    
    # 1. 加载测试数据
    test_data_path = project_root / "data" / "syn_test.json"
    print(f"\n[1/6] 加载测试数据: {test_data_path}")
    
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    if TEST_LIMIT:
        test_cases = test_cases[:TEST_LIMIT]
        print(f"      测试前 {len(test_cases)} 个案例")
    else:
        print(f"      共 {len(test_cases)} 个测试案例")
    
    # 2. 初始化 pipeline
    print("\n[2/6] 初始化 MoodAngelsPipeline")
    pipe = MoodAngelsPipeline(
        retrieval_data=project_root / "data" / "syn_train.json",
        use_llm=USE_LLM,
        llm_api_key=DEEPSEEK_API_KEY if USE_LLM else None
    )
    print("      初始化完成")
    
    # 3. 统计真实标签分布
    print("\n[3/6] 真实标签统计:")
    label_0 = sum(1 for case in test_cases if case["mood_disorder"] == 0)
    label_1 = sum(1 for case in test_cases if case["mood_disorder"] == 1)
    print(f"      无情绪障碍 (0): {label_0}")
    print(f"      有情绪障碍 (1): {label_1}")
    
    # 4. 运行批量测试
    print("\n[4/6] 运行批量诊断...")
    y_true = []
    y_pred = []
    predictions = []
    errors = []
    
    for i, case in enumerate(test_cases, 1):
        case_id = case.get("id", f"case_{i}")
        true_label = case["mood_disorder"]
        
        try:
            result = pipe.diagnose_dict(case, agent="multi")
            pred_label = result.label
            
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
                print(f"             ACC: {current_metrics['acc']*100:.1f}% | MCC: {current_metrics['mcc']:.3f}")
                
        except Exception as e:
            print(f"      案例 {case_id} 出错: {e}")
            errors.append({
                "id": case_id,
                "error": str(e)
            })
    
    # 5. 计算最终指标
    print("\n[5/6] 计算最终指标...")
    metrics = calculate_metrics(y_true, y_pred)
    cm = metrics["confusion_matrix"]
    
    # 6. 输出结果
    print("\n[6/6] 测试结果汇总:")
    print("-" * 80)
    print(f"总案例数: {len(y_true)}")
    print(f"正确数: {cm['tp'] + cm['tn']}")
    print(f"错误数: {cm['fp'] + cm['fn']}")
    print("-" * 80)
    print("\n📊 核心指标:")
    print(f"  ACC (准确率):     {metrics['acc']*100:.2f}%")
    print(f"  MCC (马修斯系数): {metrics['mcc']:.4f}")
    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    
    print("\n📈 类别详细指标:")
    print(f"  类别 0 (无情绪障碍):")
    print(f"    Precision: {metrics['class_0']['precision']:.4f}")
    print(f"    Recall:    {metrics['class_0']['recall']:.4f}")
    print(f"    F1:        {metrics['class_0']['f1']:.4f}")
    print(f"  类别 1 (有情绪障碍):")
    print(f"    Precision: {metrics['class_1']['precision']:.4f}")
    print(f"    Recall:    {metrics['class_1']['recall']:.4f}")
    print(f"    F1:        {metrics['class_1']['f1']:.4f}")
    
    print("\n🧮 混淆矩阵:")
    print(f"                  预测为 0    预测为 1")
    print(f"  真实为 0        {cm['tn']:4d}        {cm['fp']:4d}")
    print(f"  真实为 1        {cm['fn']:4d}        {cm['tp']:4d}")
    
    wrong_cases = [p for p in predictions if p["true"] != p["pred"]]
    if wrong_cases:
        print(f"\n❌ 错误案例 ({len(wrong_cases)} 个):")
        for wc in wrong_cases[:10]:
            print(f"  {wc['id']}: 真实={wc['true']}, 预测={wc['pred']}")
        if len(wrong_cases) > 10:
            print(f"  ... 还有 {len(wrong_cases) - 10} 个")
    
    output_path = project_root / "tests" / f"test_results_{'llm' if USE_LLM else 'rule'}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "use_llm": USE_LLM,
            "total": len(y_true),
            "metrics": metrics,
            "predictions": predictions,
            "errors": errors
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存到: {output_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        import pydantic
        import pandas
        import numpy
        import openai
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("\n请先运行:")
        print("  conda activate llm")
        print("  pip install pydantic pandas numpy openai")
        sys.exit(1)
    
    main()

