from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ====== 配置 ======
API_KEY = "你的key"
BASE_URL = "https://api.deepseek.com/v1"
PDF_PATH = "你的PDF文件路径"  # 随便找个PDF，比如下载一篇技术文章

# 大模型和向量模型（用同一个API的embedding）
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.1
)

# DeepSeek目前不提供embedding模型，用OpenAI兼容的免费方案
# 这里用HuggingFace的本地embedding，不需要API key
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese"  # 中文embedding模型，第一次会自动下载
)

# ====== 第一步：加载PDF，切成小块 ======
print("1. 加载PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"   共加载 {len(documents)} 页")

print("2. 切分文本...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块500字
    chunk_overlap=50,      # 块之间重叠50字，防止一句话被切断
)
chunks = text_splitter.split_documents(documents)
print(f"   切分成 {len(chunks)} 个文本块")

# ====== 第二步：向量化，存入向量数据库 ======
print("3. 向量化并存入数据库...")
vector_store = FAISS.from_documents(chunks, embeddings)
print("   向量数据库创建完成")

# ====== 第三步：创建检索器 ======
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}  # 每次检索最相关的3个文本块
)

# ====== 第四步：构建RAG链 ======
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个问答助手。请根据以下资料回答问题。如果资料中没有相关信息，请说'资料中未找到相关信息'，不要编造。\n\n资料：\n{context}"),
    ("user", "{question}")
])

def format_docs(docs):
    """把检索到的文档拼接成一段文本"""
    return "\n\n".join(doc.page_content for doc in docs)

# 组装RAG链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ====== 第五步：提问 ======
print("\n4. RAG系统就绪，开始提问...\n")

questions = [
    "这篇文档主要讲了什么？",
    "文档中提到了哪些关键技术？",
]

for q in questions:
    print(f"问: {q}")
    answer = rag_chain.invoke(q)
    print(f"答: {answer}\n")