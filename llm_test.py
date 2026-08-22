import os

import requests

API_KEY = os.getenv("DEEPSEEK_API_KEY")


def chat(prompt):
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个Java后端面试官，回答要简短、专业"},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 100,
        },
    )
    return response.json()["choices"][0]["message"]["content"]