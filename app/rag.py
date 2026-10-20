"""RAG 核心：先检索，再让大模型基于检索结果回答。

这是整个项目的灵魂，只有两个动作：
    检索：问题 → 向量 → 在 Chroma 里找最相似的 K 块
    生成：把 K 块原文 + 问题一起交给大模型，要求它按资料作答并标出处

不用 LangChain 的黑盒链式封装，每一步都看得见，方便调试，面试也好讲。
"""

from functools import lru_cache
from typing import Iterator

import chromadb
from app import config  # noqa: F401  必须第一个导入：设置 HF 镜像地址
from fastembed import TextEmbedding
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ---------------- 提示词：RAG 效果好不好，一半靠这里 ----------------
SYSTEM_PROMPT = """你是一位严谨的上市公司年报分析助手。请严格遵守以下规则：

1. 只依据「参考资料」中的内容回答，绝对不要用你自己的记忆或常识补充。
2. 每一个结论后面都要标注出处，格式为 [公司名 第X页]。
3. 如果参考资料里没有相关信息，直接回答「年报中没有找到相关说明」，不要猜测。
4. 涉及金额、比例、增长率等数字时，必须原样引用资料中的数字，不要换算或估算。
5. 用中文回答，条理清晰，不要啰嗦。"""

USER_PROMPT = """参考资料：
{context}

问题：{question}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
)


# ---------------- 单例：模型和数据库只加载一次 ----------------
# lru_cache 的作用：同一个进程里反复调用只真正执行一次。
# 向量模型加载要好几秒，每次提问都重新加载会让程序慢得没法用。
@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    """加载本地中文向量模型（必须和建库时用的是同一个模型）。"""
    return TextEmbedding(model_name=config.EMBEDDING_MODEL, cache_dir=str(config.MODEL_DIR))


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    """连接 Chroma 客户端（创建开销较大，所以缓存）。"""
    if not config.CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"向量库不存在：{config.CHROMA_DIR}\n请先运行：python -m app.ingest"
        )
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection():
    """获取集合对象。

    注意这里**刻意没有**加 @lru_cache，这是个踩过坑才明白的细节：

        重建向量库时，`app/ingest.py` 会把旧集合删掉、再建一个新集合。
        如果这里缓存了集合对象，正在运行的后端就会一直持有一个
        "已经被删除的集合"，表现为重建之后所有接口都报错，
        必须重启后端才恢复。

        只缓存 client（创建开销大），集合对象每次现取（开销很小），
        这样重建之后不用重启服务，下一次请求就自动用上新集合。
    """
    return get_client().get_collection(name=config.COLLECTION_NAME)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """连接 DeepSeek。它兼容 OpenAI 接口协议，所以能直接用 ChatOpenAI。"""
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError(
            "没有配置 DEEPSEEK_API_KEY。\n"
            "请把项目根目录的 .env 文件里的 DEEPSEEK_API_KEY 填上你的密钥。"
        )
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0,          # 年报问答要准确不要创意，所以温度设 0
    )


# ---------------- 检索 ----------------
def _retrieve_vector(question: str, k: int) -> list[dict]:
    """纯向量检索（基线方案）。"""
    embedder = get_embedder()
    collection = get_collection()

    query_vector = next(iter(embedder.embed([question]))).tolist()
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma 返回的是"批次"结构（外层列表对应每个查询），我们只查了一个问题，取 [0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        {
            "text": doc,
            "company": meta.get("company", "未知公司"),
            "page": meta.get("page"),
            "file": meta.get("file"),
            "distance": round(float(dist), 4),
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def retrieve(question: str, k: int | None = None) -> list[dict]:
    """检索最相关的 K 块，返回按相关度排序的列表（distance 越小越相关）。

    支持三种模式，由 .env 里的 RETRIEVAL_MODE 控制：
        vector        —— 纯向量检索
        hybrid        —— 向量 + BM25 关键词检索，用 RRF 融合
        hybrid_rerank —— 上面基础上再加交叉编码器精排

    做成可切换的，是为了能跑消融实验：同样的题目跑三遍，
    对比 Recall@k 到底提升了多少。没有对比就没有说服力。
    """
    k = k or config.TOP_K
    if config.RETRIEVAL_MODE in ("hybrid", "hybrid_rerank"):
        # 延迟导入：混合模式才需要 jieba/rank_bm25，没必要让所有人等
        from app.retriever import hybrid_search

        return hybrid_search(question, k=k)
    return _retrieve_vector(question, k)


def format_context(hits: list[dict]) -> str:
    """把检索结果拼成给大模型看的参考资料，并带上出处标记。"""
    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"【资料{index}｜{hit['company']} 第{hit['page']}页】\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def to_sources(hits: list[dict]) -> list[dict]:
    """把检索结果转成前端好显示的出处列表。"""
    return [
        {
            "company": hit["company"],
            "page": hit["page"],
            "file": hit["file"],
            "distance": hit["distance"],
            "preview": hit["text"][:120],
        }
        for hit in hits
    ]


# ---------------- 生成 ----------------
def answer(question: str, k: int | None = None) -> dict:
    """完整问一次：检索 → 生成。返回答案和出处（非流式）。"""
    hits = retrieve(question, k)
    chain = PROMPT | get_llm() | StrOutputParser()
    text = chain.invoke({"context": format_context(hits), "question": question})
    return {"answer": text, "sources": to_sources(hits)}


def stream_answer(question: str, k: int | None = None) -> tuple[list[dict], Iterator[str]]:
    """流式版本：先取出处列表，再返回逐字输出的生成器。

    为什么先返回出处？
        检索是瞬时的，生成要好几秒。先把出处交给前端，用户立刻能看到
        "找到了哪些资料"，体验上不会觉得卡住。
    """
    hits = retrieve(question, k)
    chain = PROMPT | get_llm() | StrOutputParser()
    stream = chain.stream({"context": format_context(hits), "question": question})
    return to_sources(hits), stream
