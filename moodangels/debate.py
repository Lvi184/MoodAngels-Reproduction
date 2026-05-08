
def debate(r, d, c, llm):
    """
    辩论阶段：
    r, d, c 是三个医生的 DiagnosisResult 对象
    """
    # 构建正方和反方的初始论点
    positive_agents = [a for a in [r, d, c] if a.label == 1]
    negative_agents = [a for a in [r, d, c] if a.label == 0]
    
    pro_summary = "\n".join([str(a) for a in positive_agents]) if positive_agents else "无"
    con_summary = "\n".join([str(a) for a in negative_agents]) if negative_agents else "无"
    
    # 多轮辩论（最多 3 轮）
    debate_history = []
    
    for round_num in range(3):
        # 正方发言
        pro_prompt = f"""
你是正方辩手，认为患者有情绪障碍。
请基于以下医生的诊断结果进行辩论：
{pro_summary}

当前辩论历史：
{chr(10).join(debate_history)}

请给出你的论点（300字以内）：
"""
        pro_reply = llm(pro_prompt)
        debate_history.append(f"[正方]: {pro_reply}")
        
        # 反方发言
        con_prompt = f"""
你是反方辩手，认为患者无情绪障碍。
请基于以下医生的诊断结果进行辩论：
{con_summary}

当前辩论历史：
{chr(10).join(debate_history)}

请给出你的论点（300字以内）：
"""
        con_reply = llm(con_prompt)
        debate_history.append(f"[反方]: {con_reply}")
        
        # 判断是否继续辩论
        judge_prompt = f"""
当前辩论历史：
{chr(10).join(debate_history)}

是否需要继续辩论？请只回答 yes 或 no。
"""
        judge_decision = llm(judge_prompt).lower().strip()
        
        if "no" in judge_decision:
            break
    
    return "\n".join(debate_history)
