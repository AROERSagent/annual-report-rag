"""混合检索：向量检索 + BM25 关键词检索，用 RRF 融合。

为什么要做这个？
    基线评测暴露了明确问题（30 题，见 README）：

        Recall@1 = 60.0%    Recall@3 = 60.0%    Recall@5 = 65.0%

    注意 Recall@1 和 Recall@3 完全一样 —— 说明检索结果要么排第 1，
    要么前 5 名里根本没有，没有"排第 2、第 3"的中间情况。

    失败案例都是同一种类型：
        "美的集团账龄超过一年的【其他应付款】是多少？"
        → 检索回来的全是"应付账款"的账龄信息，就是没有"其他应付款"。
        "招商银行的【不良贷款率】是多少？"
        → 检索不到，模型只能回答"没找到"。

    根因：向量模型对"精确词"不够敏感。它把"其他应付款"和"应付账款"
    当成语义相近的东西（确实相近），但在年报问答里这两个是**不同的会计科目**，
    必须精确区分。而年报里全是这类词：不良贷款率、毛利率、归属于母公司股东的净利润…

    BM25 恰好相反：它只看字面词频，对"其他应付款"这种精确词极准，
    但不懂语义（换个说法就废了）。

    两者互补，所以合起来用。

RRF（Reciprocal Rank Fusion）怎么融合？
    不看两边的分数（向量距离和 BM25 分数根本不是一个量纲，没法直接比），
    只看**排名**：

        某个文档的最终得分 = Σ  1 / (K + 它在第 r 个榜单里的排名)

    比如 K=60，某文档在向量榜单排第 1、在 BM25 榜单排第 5：
        得分 = 1/(60+1) + 1/(60+5) = 0.01639 + 0.01538 = 0.03177

    这个方法的优点：不需要调权重、对两边的分数量纲不敏感，是工业界常用做法。
"""

from functools import lru_cache

import jieba
from app import config
from app.rag import get_collection, get_embedder

# jieba 第一次分词会加载词典，先初始化避免第一次调用卡顿
jieba.initialize()

# 缓存：向量库里的全部文本块 + 建好的 BM25 索引
# 建索引要分词 4000 多块，只在进程内做一次
_bm25_state: dict | None = None


def tokenize(text: str) -> list[str]:
    """中文分词，BM25 需要词序列而不是整句。

    用 cut_for_search 模式：它会把长词再切出细粒度的子词
    （比如"不良贷款率" → 不良/贷款/率/不良贷款/贷款率），
    对检索更友好 —— 用户问"不良贷款"也能匹配到"不良贷款率"。
    """
    return [t.strip() for t in jieba.cut_for_search(text) if t.strip()]


def get_bm25_state() -> dict:
    """构建（或读取缓存的）BM25 索引。"""
    global _bm25_state
    if _bm25_state is not None:
        return _bm25_state

    collection = get_collection()
    data = collection.get(include=["documents", "metadatas"])
    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]

    print(f"[混合检索] 正在为 {len(docs)} 个文本块建立 BM25 索引（首次调用需要一点时间）...")
    from rank_bm25 import BM25Okapi

    corpus = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(corpus)

    # id -> (正文, 元信息) 的查表，后面融合时要用
    lookup = {i: (d, m) for i, d, m in zip(ids, docs, metas)}
    _bm25_state = {"ids": ids, "bm25": bm25, "lookup": lookup}
    print("[混合检索] BM25 索引就绪")
    return _bm25_state


@lru_cache(maxsize=1)
def get_reranker():
    """加载交叉编码器重排模型。

    为什么需要它？
        RRF 融合只是"粗排"——它只看两个榜单的排名，并不真正读内容。
        实测发现一个典型失败：

            问"比亚迪的CEO是谁"，"比亚迪"这个词在库里出现在 795 个块里，
            BM25 于是返回一大堆"真·比亚迪但页码不对"的块。
            RRF 把这些块排上去，反而把向量检索本来排第 4 的正确页挤出了前 5。

        根因：RRF 认为"BM25 排第 1"和"向量排第 1"一样可信，
        但它无法判断 BM25 那个第 1 其实是靠一个没有区分度的公司名混上来的。

        交叉编码器重排解决这个问题：它把「问题 + 候选原文」拼在一起送进模型，
        真正读一遍算相关性分数。贵（比向量检索慢很多），但只对 20 个候选算，
        成本可控。这是标准的"召回粗排 + 重排精排"两阶段架构。
    """
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=config.RERANK_MODEL, cache_dir=str(config.MODEL_DIR))


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """算余弦距离（1 - 余弦相似度），和 Chroma 的 cosine 空间保持一致。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def hybrid_search(question: str, k: int, candidate: int | None = None) -> list[dict]:
    """混合检索主函数，返回格式和 app.rag.retrieve() 完全一致。"""
    candidate = candidate or config.CANDIDATE_K
    state = get_bm25_state()
    collection = get_collection()

    # ---------- ① 向量检索：先用较大的 candidate 池子，不要直接取 k ----------
    query_vector = next(iter(get_embedder().embed([question]))).tolist()
    vector_result = collection.query(
        query_embeddings=[query_vector],
        n_results=candidate,
        include=["documents", "metadatas", "distances"],
    )
    vector_ids = vector_result["ids"][0]

    # ---------- ② BM25 关键词检索 ----------
    scores = state["bm25"].get_scores(tokenize(question))
    # 关键细节：只保留真正匹配到关键词的候选（分数 > 0）。
    #
    # 如果问题里的词在库里一个都没有（典型例子：问"比亚迪的CEO是谁"，
    # 而年报里只用"总裁"这个说法，没有 "CEO" 三个字母），BM25 会给所有块
    # 打 0 分。把这 20 个 0 分块塞进融合，等于往好结果里掺噪音 ——
    # 实测会把向量检索本来排第 4 的正确页挤出前 5 名。
    #
    # 所以宁可少给候选，也不要给垃圾候选。
    ranked_idx = sorted(range(len(scores)), key=lambda i: -scores[i])
    top_idx = [i for i in ranked_idx if scores[i] > 0][:candidate]
    bm25_ids = [state["ids"][i] for i in top_idx]

    # ---------- ③ RRF 融合 ----------
    # 只看排名不看分数，所以两个榜单的量纲差异无所谓
    fused: dict[str, float] = {}
    for ranking in (vector_ids, bm25_ids):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (config.RRF_K + rank)

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])

    # ---------- ④ 交叉编码器重排（精排，可选）----------
    # 把 RRF 粗排出来的前 N 个候选，逐个和问题一起送进交叉编码器，
    # 让它真正"读一遍"再打分。只对 N 个候选算，所以成本可控。
    if config.RETRIEVAL_MODE == "hybrid_rerank" and ordered:
        pool = ordered[: config.RERANK_CANDIDATES]
        pool_ids = [doc_id for doc_id, _ in pool]
        pool_texts = [state["lookup"][doc_id][0] for doc_id in pool_ids]

        scores = list(get_reranker().rerank(question, pool_texts))
        score_by_id = dict(zip(pool_ids, scores))
        # 交叉编码器分数越大越相关
        ordered = sorted(pool, key=lambda kv: -score_by_id.get(kv[0], 0.0))

    ordered = ordered[:k]

    # ---------- ⑤ 组装返回值 ----------
    # distance 字段用真实余弦距离填充（从向量库里取回原始向量现算），
    # 这样前端和评测脚本的显示逻辑不用改。
    hit_ids = [doc_id for doc_id, _ in ordered]
    stored = collection.get(ids=hit_ids, include=["embeddings"])
    embedding_by_id = {
        i: e for i, e in zip(stored["ids"], stored["embeddings"])
    }

    hits: list[dict] = []
    for doc_id, rrf_score in ordered:
        doc, meta = state["lookup"][doc_id]
        vector = embedding_by_id.get(doc_id)
        distance = (
            round(_cosine_distance(query_vector, list(vector)), 4) if vector is not None else None
        )
        hits.append(
            {
                "text": doc,
                "company": meta.get("company", "未知公司"),
                "page": meta.get("page"),
                "file": meta.get("file"),
                "distance": distance,
                "rrf_score": round(rrf_score, 5),
            }
        )
    return hits
