import requests
import json

# 你的API Key
API_KEY = "sk-8f022f7d9f12435bb8cecce6a0aa9829"
TOOLS = """
你可以使用以下工具：

1. calculator(expression)
   功能：计算数学表达式
   参数：expression - 数学表达式字符串，比如 "123 + 456"
   用法：如果需要计算，输出 JSON: {"tool": "calculator", "expression": "123 + 456"}
"""

def calcaulator(expression):
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算出错: {e}"

















# 调大模型API
def chat(prompt):
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个Java后端面试官，回答要简短、专业"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100
        }
    )
    data = response.json()
    return data["choices"][0]["message"]["content"]

