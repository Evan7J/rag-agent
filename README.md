# RAG Agent · 文档问答

一个把 RAG 和 Function Calling 从底层到框架走了一遍的练习项目。用一份 PDF 做知识库，支持自然语言问答。

写这个项目的目的是搞清楚一件事：**RAG 到底是怎么工作的，以及 Agent 里的"工具调用"到底发生了什么。** 所以代码不是一步到位，而是按理解顺序一层层搭起来的——从裸调 API 开始，到用框架，最后再回到不用框架手写一遍。

## 快速开始

```bash
pip install -r requirements.txt

# 设置 DeepSeek API Key
# PowerShell:
$env:DEEPSEEK_API_KEY="你的key"
# Bash:
export DEEPSEEK_API_KEY="你的key"

python api_server.py
```

服务跑在 `http://localhost:8000`。

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"文档里提到了哪些技术？"}'
```

> 需要把待检索的 PDF 放到项目根目录并命名为 `resume.pdf`（在 `api_server.py` / `rag.py` / `agent_rag.py` 里改 `PDF_PATH` 也可以）。PDF 本身不进版本库。

## 技术栈

- Python 3.10+
- LangChain + LangGraph（`create_react_agent`）
- FastAPI + Uvicorn
- FAISS 向量库
- HuggingFace `text2vec-base-chinese`（本地 Embedding，不依赖外部 API）
- DeepSeek `deepseek-chat`（LLM）

## 文件说明

这个仓库按"理解路径"组织，每个文件解决一个阶段的问题：

| 文件 | 阶段 | 干了什么 |
|------|------|---------|
| `llm_test.py` | 0 | 先确认 API 能通，跑通一次最基础的对话调用 |
| `rag.py` | 1 | 基础 RAG 链：PDF → 切分 → 向量化 → 检索 → 拼 prompt → 回答 |
| `agent_rag.py` | 2 | 把检索包装成 tool，交给 LangGraph 的 ReAct Agent，由模型自主决定要不要查文档 |
| `api_server.py` | 3 | 把 Agent 包成 FastAPI 服务，对外提供 `/ask` 接口 |
| `function_calling_native.py` | 4 | **不依赖 LangChain**，手写 tool schema、手动解析 `tool_calls`、自己维护消息循环 |

### 为什么第 4 个文件要手写一遍

用 LangChain 的时候，`@tool` 装饰器一加、`create_react_agent` 一调，工具调用就跑起来了。但框架把很多东西藏起来了：

- tool schema 是自动从函数签名和 docstring 生成的
- 模型返回的 `tool_calls` 是框架解析的
- 工具执行结果怎么回填成 `tool` message 也是框架做的
- 循环什么时候结束也是框架判断的

不看一眼底层，很容易变成"只会调框架"。所以最后单独写了一份不依赖框架的实现，把整个流程摊开：

```
用户提问
  → 带上 tools schema 调 LLM
  → LLM 返回 tool_calls（模型决定调哪个工具、参数是什么）
  → 本地执行对应函数
  → 把结果作为 role=tool 的消息塞回消息列表
  → 再次调 LLM
  → 直到模型不再返回 tool_calls，输出最终答案
```

这个循环走通之后，再回头看 LangGraph 的 ReAct Agent，就能看清楚它到底帮忙做了哪些事。

## 两个关键设计点

### 1. 检索作为工具，而不是固定流程

`rag.py` 里是"每次都检索，然后把结果塞进 prompt"，属于固定流程。

`agent_rag.py` 改成了"把检索包装成工具，让模型自己决定要不要用"。区别在测试用例里能直接看出来：

```
用户: 1加1等于几？              → 不需要查文档，模型直接回答
用户: 这篇文档主要讲了什么？      → 需要查文档，模型调用 search_document
```

如果所有问题都无脑检索一遍，既浪费算力，也会因为塞入无关内容而干扰回答。让模型自己判断，是 Agent 和固定 RAG 链最本质的区别。

### 2. Embedding 用本地模型

DeepSeek 没有提供 Embedding 接口，所以用了 HuggingFace 的 `shibing624/text2vec-base-chinese`——中文场景效果不错，而且是本地跑的，不需要额外 API key，也不产生调用费用。第一次运行会自动下载模型权重。

## 踩过的坑

1. **切分粒度**：`chunk_size` 一开始设得太大（1000+），检索回来的块里混了太多无关内容，模型容易被干扰。改成 500 并加 50 的 overlap，效果明显变好。
2. **没有"不知道"的出口**：最初的 prompt 没约束，资料里没有的内容模型也会编。加了"资料中未找到相关信息，不要编造"之后，幻觉少了很多。
3. **以为框架是必须的**：用 LangChain 很顺手，但写 `function_calling_native.py` 的时候才发现，框架做的其实就是那几步——手写一遍之后，对 Agent 的理解才真正落地。

## 后续可以做的

- [ ] 支持多文档 / 整个目录作为知识库
- [ ] 加 rerank（检索后重排），提升召回精度
- [ ] 向量库持久化到磁盘，避免每次启动重建
- [ ] 加对话历史，支持多轮追问
- [ ] 补充检索效果的评测（召回率 / 命中率）

## License

MIT