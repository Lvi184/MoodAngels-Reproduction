
#!/usr/bin/env python3
"""
批量测试脚本 - 直接运行，不需要 pip install -e
测试 syn_test.json 中的所有案例
包含 ACC、MCC、Macro F1 等指标
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
    # 混淆矩阵
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    # 准确率 (ACC)
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    
    # MCC (Matthews Correlation Coefficient)
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom != 0 else 0
    
    # 各类别的 Precision, Recall, F1
    # 类别 0
    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f10 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0
    
    # 类别 1
    prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f11 = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) > 0 else 0
    
    # Macro F1
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
    print("=" * 70)
    print("MoodAngels 批量测试 - 含 ACC/MCC/Macro F1")
    print("=" * 70)
    
    # 1. 加载测试数据
    test_data_path = project_root / "data" / "syn_test.json"
    print(f"\n[1/6] 加载测试数据: {test_data_path}")
    
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    print(f"      共加载 {len(test_cases)} 个测试案例")
    
    # 2. 初始化 pipeline
    print("\n[2/6] 初始化 MoodAngelsPipeline")
    pipe = MoodAngelsPipeline(project_root / "data" / "syn_train.json")
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
                # 计算当前指标
                current_metrics = calculate_metrics(y_true, y_pred)
                print(f"      进度: {i}/{len(test_cases)} ({100*i/len(test_cases):.1f}%)")
                print(f"             ACC: {current_metrics['acc']*100:.1f}% | MCC: {current_metrics['mcc']:.3f} | Macro F1: {current_metrics['macro_f1']:.3f}")
                
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
    print("-" * 70)
    print(f"总案例数: {len(y_true)}")
    print(f"正确数: {cm['tp'] + cm['tn']}")
    print(f"错误数: {cm['fp'] + cm['fn']}")
    print("-" * 70)
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
    
    # 详细错误分析
    wrong_cases = [p for p in predictions if p["true"] != p["pred"]]
    if wrong_cases:
        print("\n❌ 错误案例详情:")
        for wc in wrong_cases:
            print(f"  {wc['id']}: 真实={wc['true']}, 预测={wc['pred']}")
    
    # 保存结果
    output_path = project_root / "tests" / "test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(y_true),
            "metrics": metrics,
            "predictions": predictions,
            "errors": errors
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存到: {output_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # 检查依赖
    try:
        import pydantic
        import pandas
        import numpy
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("\n请先运行:")
        print("  conda activate llm")
        print("  pip install pydantic pandas numpy")
        sys.exit(1)
    
    main()

