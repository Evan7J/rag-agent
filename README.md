# RAG智能问答Agent

基于 LangChain + FastAPI 的 RAG 文档问答系统，支持 PDF 文档检索与智能问答。

## 技术栈

- Python 3.10+
- LangChain + LangGraph
- FastAPI
- FAISS 向量数据库
- DeepSeek 大模型
- HuggingFace Embedding

## 快速启动

```bash
# 设置 DeepSeek API Key（Windows PowerShell）
$env:DEEPSEEK_API_KEY="你的key"

pip install -r requirements.txt
python api_server.py
```

## 接口

- `POST /ask` 提问，body 传 `{"question": "..."}`
- `GET /health` 健康检查

> 把要检索的 PDF 放到项目根目录，重命名为 `resume.pdf`，或改 `api_server.py` 里的 `PDF_PATH`。