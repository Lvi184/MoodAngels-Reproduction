
#!/usr/bin/env python3
"""
DeepSeek API 客户端
用于调用真实的 LLM API
"""

import os
import json
from openai import OpenAI


class DeepSeekClient:
    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com"):
        """
        初始化 DeepSeek 客户端
        
        Args:
            api_key: DeepSeek API Key，如果为 None 则从环境变量 DEEPSEEK_API_KEY 读取
            base_url: API 基础 URL
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供 DeepSeek API Key，可以通过参数或环境变量 DEEPSEEK_API_KEY 设置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        self.model = "deepseek-chat"
    
    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        调用 DeepSeek 聊天 API
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
            temperature: 温度参数
            max_tokens: 最大生成 token 数
        
        Returns:
            模型返回的文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API 调用错误: {e}")
            return ""
    
    def chat_with_json(self, messages: list, temperature: float = 0.3, max_tokens: int = 2000) -> dict:
        """
        调用 DeepSeek API 并尝试解析 JSON 响应
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
        
        Returns:
            解析后的 JSON 对象
        """
        content = self.chat(messages, temperature, max_tokens)
        if not content:
            return {}
        
        # 尝试提取 JSON
        try:
            # 找到第一个 { 和最后一个 }
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                return json.loads(json_str)
            else:
                return {}
        except Exception as e:
            print(f"JSON 解析错误: {e}")
            return {}

