
#!/usr/bin/env python3
"""直接测试 - 不需要 pip install -e"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from moodangels.pipeline import MoodAngelsPipeline
from moodangels.data import load_cases

def test_direct():
    print("=== 直接 Python 测试 (无需 pip install -e) ===\n")
    
    # 数据路径
    data_path = project_root / "data" / "syn_train.json"
    
    # 1. 加载案例
    print("1. 加载案例数据...")
    cases = load_cases(str(data_path))
    print(f"   加载了 {len(cases)} 个案例\n")
    
    # 2. 初始化 Pipeline
    print("2. 初始化 MoodAngelsPipeline...")
    pipe = MoodAngelsPipeline(str(data_path))
    print("   初始化完成\n")
    
    # 3. 测试诊断
    print("3. 运行诊断测试...")
    test_case = cases[0]
    result = pipe.diagnose_dict(test_case.raw, agent="multi")
    
    # 4. 输出结果
    print("4. 结果:")
    print(f"   label = {result.label}")
    print(f"   单智能体结果存在: {bool(result.single_agent_results)}")
    
    # 验证
    assert result.label in (0, 1), "标签应是 0 或 1"
    assert result.single_agent_results, "应该有单智能体结果"
    
    print("\n✅ 测试成功！")

if __name__ == "__main__":
    # 检查依赖是否安装
    try:
        import pydantic
        import pandas
        import numpy
        print("依赖检查：✓ pydantic, pandas, numpy 已安装\n")
    except ImportError as e:
        print(f"⚠️  缺少依赖: {e}")
        print("请先运行: pip install pydantic pandas numpy\n")
        sys.exit(1)
    
    test_direct()
