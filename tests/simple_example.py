
#!/usr/bin/env python3
"""最简单的 MoodAngels 使用示例"""

import sys
sys.path.insert(0, '.')  # 确保能找到 moodangels 模块

from moodangels.pipeline import MoodAngelsPipeline

# 创建一个简单的测试用例文本
test_text = """
患者自述：最近一个月心情一直很低落，对什么都提不起兴趣，
晚上经常失眠，早上很早就醒了，觉得自己很没用，
有时候会想活着没什么意思。
"""

# 初始化 pipeline
pipe = MoodAngelsPipeline('data/syn_train.json')

# 进行诊断
result = pipe.diagnose_dict(test_text, agent='multi')

# 打印结果
print("诊断结果：")
print(f"标签：{result.label}")
print("\n详细信息：")
print(result)
