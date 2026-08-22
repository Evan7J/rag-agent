"""原生 Function Calling 实现，不依赖 LangChain。"""

import json
import os

from openai import OpenAI

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# 第一步：定义工具

# 工具1：计算器
calculator_schema = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "执行数学计算。支持加减乘除、乘方等运算。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，例如 '2+3*4'、'100/7'、'2**10'"
                }
            },
            "required": ["expression"]
        }
    }
}

# 工具2：获取天气（模拟）
weather_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的当前天气。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，例如 '北京'、'上海'、'深圳'"
                }
            },
            "required": ["city"]
        }
    }
}

# 所有工具汇总
TOOLS = [calculator_schema, weather_schema]


# 第二步：函数实现

def calculator(expression: str) -> str:
    """安全执行数学表达式"""
    try:
        allowed = set("0123456789+-*/().% ^")
        if not set(expression).issubset(allowed):
            return "错误：表达式包含不允许的字符"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


def get_weather(city: str) -> str:
    """模拟天气查询"""
    weather_data = {
        "北京": "晴天，26°C，湿度40%",
        "上海": "多云，30°C，湿度70%",
        "深圳": "阵雨，28°C，湿度85%",
        "杭州": "阴天，25°C，湿度60%",
    }
    return weather_data.get(city, f"暂无{city}的天气数据，请尝试其他城市")


# 函数名 -> 实际函数的映射
FUNCTION_MAP = {
    "calculator": calculator,
    "get_weather": get_weather,
}


# 第三步：核心循环

def run_agent(user_question: str, max_turns: int = 10) -> str:
    # 消息列表，初始只有用户问题
    messages = [{"role": "user", "content": user_question}]

    print(f"\n{'=' * 60}")
    print(f"用户: {user_question}")
    print(f"{'=' * 60}")

    for turn in range(1, max_turns + 1):
        # 3.1 调用 LLM
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            temperature=0.1,
        )

        msg = response.choices[0].message

        # 3.2 LLM 要调工具
        if msg.tool_calls:
            messages.append(msg)

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"  [第{turn}轮] LLM 调用工具: {func_name}({func_args})")

                func = FUNCTION_MAP.get(func_name)
                if func:
                    result = func(**func_args)
                else:
                    result = f"错误：未知工具 {func_name}"

                print(f"  [第{turn}轮] 工具返回: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            continue

        # 3.3 返回普通文本，对话结束
        if msg.content:
            print(f"  [第{turn}轮] LLM 最终回复")
            return msg.content

        return "（LLM 没有返回任何内容）"

    return f"已达最大轮次 {max_turns}，强制终止"


# 第四步：流式版本

def run_agent_stream(user_question: str, max_turns: int = 10) -> str:
    # tool_calls 在流式下分片返回，先拼接完整再判断
    messages = [{"role": "user", "content": user_question}]

    print(f"\n{'=' * 60}")
    print(f"用户: {user_question}")
    print(f"{'=' * 60}")

    for turn in range(1, max_turns + 1):
        # 4.1 流式调用
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            stream=True,
        )

        # 4.2 收集流式片段
        content_parts = []
        tool_call_buf = {}

        for chunk in stream:
            delta = chunk.choices[0].delta

            # 普通文字片段，逐字打印
            if delta.content:
                content_parts.append(delta.content)
                print(delta.content, end="", flush=True)

            # tool_calls 片段，需要拼接
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_buf:
                        tool_call_buf[idx] = {
                            "id": "",
                            "function_name": "",
                            "function_args": ""
                        }
                    buf = tool_call_buf[idx]
                    if tc.id:
                        buf["id"] += tc.id
                    if tc.function:
                        if tc.function.name:
                            buf["function_name"] += tc.function.name
                        if tc.function.arguments:
                            buf["function_args"] += tc.function.arguments

        # 4.3 判断：调工具还是直接回复
        if tool_call_buf:
            print()

            assistant_msg = {
                "role": "assistant",
                "content": "".join(content_parts) if content_parts else None,
                "tool_calls": [
                    {
                        "id": buf["id"],
                        "type": "function",
                        "function": {
                            "name": buf["function_name"],
                            "arguments": buf["function_args"]
                        }
                    }
                    for buf in tool_call_buf.values()
                ]
            }
            messages.append(assistant_msg)

            for buf in tool_call_buf.values():
                func_name = buf["function_name"]
                func_args = json.loads(buf["function_args"])
                print(f"\n  [第{turn}轮] 调用工具: {func_name}({func_args})")

                func = FUNCTION_MAP.get(func_name)
                result = func(**func_args) if func else f"未知工具 {func_name}"
                print(f"  [第{turn}轮] 工具返回: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": buf["id"],
                    "content": result,
                })

            continue

        # 4.4 直接回复完成
        print()
        return "".join(content_parts)

    return f"已达最大轮次 {max_turns}，强制终止"


# 第五步：测试

if __name__ == "__main__":
    print("=" * 60)
    print("  测试1：普通版本（一次性返回）")
    print("=" * 60)
    answer = run_agent("北京今天天气怎么样？")
    print(f"Agent: {answer}\n")

    print("\n")
    print("=" * 60)
    print("  测试2：流式版本（逐字输出）")
    print("=" * 60)
    answer = run_agent_stream("北京今天天气怎么样？")
    print(f"\n\nAgent 完整回复: {answer}\n")

    print("\n")
    print("=" * 60)
    print("  测试3：流式 + 多工具调用")
    print("=" * 60)
    answer = run_agent_stream("计算2的10次方，然后告诉我杭州的天气")
    print(f"\n\nAgent 完整回复: {answer}")