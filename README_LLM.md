
# MoodAngels LLM 版本使用说明

## 快速开始

### 1. 安装依赖

```bash
conda activate llm
pip install pydantic pandas numpy openai
```

### 2. 运行测试

#### 规则版本（默认，不需要 API）

```bash
cd F:\muti_agent\MoodAngels_complete
python tests\batch_test_with_llm.py
```

#### LLM 版本（使用 DeepSeek API）

编辑 `tests\batch_test_with_llm.py`，修改配置：

```python
USE_LLM = True  # 开启 LLM 版本
DEEPSEEK_API_KEY = "sk-7ccd72838c414662870657b1a8a666c8"  # 你的 API Key
```

然后运行：

```bash
python tests\batch_test_with_llm.py
```

## 文件结构

```
moodangels/
├── pipeline.py          # 主 Pipeline（支持规则/LLM 切换）
├── agents.py            # 规则版本智能体
├── agents_llm.py        # LLM 版本智能体（新增）
├── llm_client.py        # DeepSeek API 客户端（新增）
├── data.py              # 数据加载
├── retrieval.py         # 检索模块
├── granular.py          # 粒度分析
└── schemas.py           # 数据结构

tests/
├── batch_test_with_llm.py  # 批量测试脚本（支持切换）
├── batch_test.py          # 旧版测试（仅规则）
└── ...
```

## 使用示例

### Python 代码中使用

#### 规则版本

```python
from moodangels.pipeline import MoodAngelsPipeline

pipe = MoodAngelsPipeline("data/syn_train.json", use_llm=False)
result = pipe.diagnose_dict(case_dict, agent="multi")
print(result.label)  # 0 或 1
```

#### LLM 版本

```python
from moodangels.pipeline import MoodAngelsPipeline

pipe = MoodAngelsPipeline(
    "data/syn_train.json", 
    use_llm=True,
    llm_api_key="sk-7ccd72838c414662870657b1a8a666c8"
)
result = pipe.diagnose_dict(case_dict, agent="multi")
print(result.label)
```

## 对比：规则 vs LLM

| 特性 | 规则版本 | LLM 版本 |
|------|---------|---------|
| 需要 API | ❌ | ✅ |
| 运行速度 | 快 | 慢（需调用 API） |
| 可解释性 | 高（公式明确） | 中（依赖 LLM） |
| 准确率 | 基准 | 可能更高 |
| 成本 | 免费 | 有 API 费用 |

## 官方代码对比

原官方 `MoodAngels-main/` 包含：
- 工具调用框架
- 辩论机制
- 完整的 prompt 设计

本版本是简化实现，核心功能保留：
- ✅ 三个智能体 (Angel.R, Angel.D, Angel.C)
- ✅ 检索增强
- ✅ 辩论 + 置信度加权
- ✅ DeepSeek API 集成

## 下一步

如需更完整的实现，可以参考官方代码进一步扩展：
- 工具调用 (retrieve_dsm5, retrieve_similar_records)
- 更复杂的 prompt 设计
- 多轮对话 + 反思机制
