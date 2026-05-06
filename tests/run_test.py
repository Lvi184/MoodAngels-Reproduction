
#!/usr/bin/env python3
"""直接用 Python 脚本运行 MoodAngels 测试"""

from moodangels.pipeline import MoodAngelsPipeline
from moodangels.data import load_cases

def main():
    print("=== MoodAngels 直接 Python 测试 ===\n")
    
    # 1. 加载案例数据
    print("1. 加载案例数据...")
    cases = load_cases('data/syn_train.json')
    print(f"   成功加载 {len(cases)} 个案例\n")
    
    # 2. 初始化 pipeline
    print("2. 初始化 MoodAngelsPipeline...")
    pipe = MoodAngelsPipeline('data/syn_train.json')
    print("   初始化完成\n")
    
    # 3. 测试第一个案例
    print("3. 测试第一个案例诊断...")
    first_case = cases[0]
    print(f"   案例内容（前200字符）: {str(first_case.raw)[:200]}...\n")
    
    # 使用 multi-agent 模式诊断
    res = pipe.diagnose_dict(first_case.raw, agent='multi')
    
    # 4. 输出结果
    print("4. 诊断结果:")
    print(f"   标签 (label): {res.label}")
    print(f"   是否有单智能体结果: {bool(res.single_agent_results)}")
    if res.single_agent_results:
        print(f"   单智能体结果数量: {len(res.single_agent_results)}")
    
    # 验证结果
    assert res.label in (0, 1), f"标签应该是 0 或 1，实际是 {res.label}"
    assert res.single_agent_results, "应该有单智能体结果"
    
    print("\n✅ 测试通过！")

if __name__ == "__main__":
    main()
