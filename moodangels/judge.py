
def judge(r, d, c, debate_result, llm):
    """
    裁判阶段：基于三个医生的诊断和辩论历史做出最终决定
    
    返回格式（JSON）：
    {
        "label": 0 或 1,
        "confidence": 0.0-1.0,
        "reasoning": "最终判断理由"
    }
    """
    judge_prompt = f"""
你是精神科诊断裁判，需要基于以下信息做出最终诊断：

三个医生的诊断结果：
- {r}
- {d}
- {c}

辩论历史：
{debate_result}

请给出你的最终诊断，严格按照以下 JSON 格式输出：
{{
    "label": 0 或 1 (0 表示无情绪障碍, 1 表示有情绪障碍),
    "confidence": 0.0 到 1.0 之间的置信度,
    "reasoning": "简短的诊断理由"
}}
"""
    
    # 调用 LLM
    result_text = llm(judge_prompt)
    
    # 尝试解析 JSON（如果失败则回退到启发式）
    try:
        import json
        # 尝试找到 JSON 部分
        start_idx = result_text.find("{")
        end_idx = result_text.rfind("}") + 1
        if start_idx != -1 and end_idx != -1:
            json_str = result_text[start_idx:end_idx]
            result = json.loads(json_str)
            # 格式化返回
            label = result.get("label", 0)
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "")
            return json.dumps({
                "label": label,
                "confidence": confidence,
                "reasoning": reasoning
            }, ensure_ascii=False)
    except Exception:
        pass
    
    # 回退：用简单的启发式 + 原文
    return result_text
