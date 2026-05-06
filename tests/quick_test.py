
#!/usr/bin/env python3
"""最简化测试 - 只需安装基础依赖"""

import sys
from pathlib import Path

# 1. 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 2. 导入
from moodangels.pipeline import MoodAngelsPipeline
from moodangels.data import load_cases

# 3. 准备数据路径
data_path = Path(__file__).parent.parent / "data" / "syn_train.json"

# 4. 加载数据并测试
cases = load_cases(str(data_path))
pipe = MoodAngelsPipeline(str(data_path))
result = pipe.diagnose_dict(cases[0].raw, agent="multi")

# 5. 输出
print(f"诊断标签: {result.label}")
print("测试完成！")
