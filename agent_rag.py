from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

API_KEY = "你的key"
PDF_PATH = "你的PDF路径"

# 大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1",
    temperature=0.1
)

# 加载PDF，建向量库（跟之前一样）
print("加载PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print(f"PDF加载完成，共 {len(chunks)} 个文本块\n")

# 关键：把检索包装成工具，Agent可以自主决定要不要调用
@tool
def search_document(query: str) -> str:
    """在PDF文档中搜索相关内容。当用户问的问题需要查文档时使用此工具。
    参数 query: 搜索关键词或问题"""
    docs = retriever.invoke(query)
    if not docs:
        return "未找到相关内容"
    return "\n\n".join(f"[来源第{d.metadata.get('page', '未知')}页] {d.page_content}" for d in docs)

# 创建Agent，把检索工具给它
agent = create_react_agent(llm, [search_document])

# 测试
questions = [
    "1加1等于几？",                       # 不需要查文档
    "这篇文档主要讲了什么？",               # 需要查文档
    "文档里有没有提到数据库相关的内容？",    # 需要查文档
]

for q in questions:
    print(f"用户: {q}")
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    answer = result["messages"][-1].content
    print(f"Agent: {answer}\n")
    print("---\n")