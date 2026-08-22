from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import uvicorn

# ====== 配置 ======
API_KEY = os.getenv("DEEPSEEK_API_KEY")
PDF_PATH = "resume.pdf"

# ====== 初始化（服务启动时执行一次） ======
print("服务启动中...")
llm = ChatOpenAI(model="deepseek-chat", api_key=API_KEY, base_url="https://api.deepseek.com/v1", temperature=0.1)

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

@tool
def search_document(query: str) -> str:
    """在PDF文档中搜索相关内容"""
    docs = retriever.invoke(query)
    if not docs:
        return "未找到相关内容"
    return "\n\n".join(f"[第{d.metadata.get('page', '?')}页] {d.page_content}" for d in docs)

agent = create_react_agent(llm, [search_document])
print("服务启动完成！\n")

# ====== API定义 ======
app = FastAPI(title="RAG Agent API")

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

@app.post("/ask", response_model=Answer)
def ask(req: Question):
    """问问题，Agent自主决定是否需要检索文档"""
    result = agent.invoke({"messages": [{"role": "user", "content": req.question}]})
    return Answer(answer=result["messages"][-1].content)

@app.get("/health")
def health():
    return {"status": "ok"}

# ====== 启动 ======
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)