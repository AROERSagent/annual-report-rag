"""FastAPI 接口层 —— 把命令行问答变成 HTTP 服务。

运行（在项目根目录下）：
    python -m app.api
    # 或者用 uvicorn（--reload 表示改代码自动重启，开发时很方便）
    uvicorn app.api:app --reload --port 8000

打开 http://127.0.0.1:8000/docs 可以看到自动生成的接口文档，能直接在网页上测试。

本文件刻意保持"薄"：只负责 HTTP 相关的三件事
    1. 接收请求、校验参数（Pydantic 模型）
    2. 调用 app/rag.py 里的核心逻辑
    3. 把结果序列化成 JSON / SSE
所有 RAG 业务逻辑都在 app/rag.py 里，这样换界面不用改逻辑。
"""

import json
import sys
from typing import AsyncIterator

import uvicorn
from app import config  # noqa: F401  必须第一个导入
from app.rag import answer, stream_answer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = FastAPI(
    title="上市公司年报 RAG 问答",
    description="基于真实 A 股年报的检索增强问答，答案带页码引用。",
    version="0.1.0",
)

# ---------- 跨域配置 ----------
# React 开发服务器跑在 3000（Create React App）或 5173（Vite）端口，
# 与 API 的 8000 端口不同源，浏览器会拦截请求，所以必须在服务端放行。
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 请求 / 响应的数据结构 ----------
# 用 Pydantic 定义的好处：FastAPI 会自动校验参数并把错误变成清晰的 422 响应，
# 不用自己写一堆 if 判断。
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="检索几块，不传则用配置里的默认值"
    )


class Source(BaseModel):
    company: str = Field(description="公司名")
    page: int | None = Field(default=None, description="页码")
    file: str | None = Field(default=None, description="来源文件")
    distance: float = Field(description="相似度距离，越小越相关")
    preview: str = Field(description="原文片段")


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


# ---------- 接口 ----------
@app.get("/", summary="健康检查")
def root() -> dict:
    """最简单的存活检查，浏览器直接打开就能确认服务在跑。"""
    return {
        "service": "annual-report-rag",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/api/info", summary="知识库信息")
def info() -> dict:
    """告诉前端知识库里有哪些公司、共多少块，方便做筛选和展示。"""
    try:
        from app.rag import get_collection

        collection = get_collection()
        data = collection.get(include=["metadatas"])
        companies = sorted({m.get("company", "") for m in data["metadatas"] if m})
        return {
            "collection": config.COLLECTION_NAME,
            "chunks": collection.count(),
            "companies": companies,
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.LLM_MODEL,
            "default_top_k": config.TOP_K,
        }
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"向量库不可用：{exc}") from exc


@app.post("/api/chat", response_model=ChatResponse, summary="问答（一次性返回）")
def chat(req: ChatRequest) -> ChatResponse:
    """先把最简单的一版做出来：等模型生成完，一次性返回完整结果。

    缺点：用户要干等 2~5 秒，期间页面上什么都不会变。
    """
    try:
        result = answer(req.question, k=req.top_k)
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"生成失败：{exc}") from exc
    return ChatResponse(**result)


@app.post("/api/chat/stream", summary="问答（SSE 流式返回）")
async def chat_stream(req: ChatRequest) -> EventSourceResponse:
    """流式版本，这是前端实际会用的接口。

    事件顺序（前端按 event 字段区分处理）：
        event: sources  → 检索到的出处列表（立刻发出，用户马上能看到"找到了哪些资料"）
        event: delta    → 模型的逐段输出（多次）
        event: error    → 出错信息
        event: done     → 结束标记

    为什么先发 sources？
        检索只要几十毫秒，生成要好几秒。先把出处推给前端，
        用户立刻看到反馈，体验上不会觉得卡住。
    """

    async def event_generator() -> AsyncIterator[dict]:
        try:
            # 注意：检索这一步是同步的（要跑向量模型），会短暂阻塞事件循环。
            # 单机演示可以接受；要上生产就得把它也改成异步或丢进线程池。
            sources, sync_stream = stream_answer(req.question, k=req.top_k)

            yield {
                "event": "sources",
                "data": json.dumps(sources, ensure_ascii=False),
            }

            # LangChain 的 .stream() 返回的是同步生成器，直接在 async 函数里 for 循环
            # 会阻塞整个事件循环（其他请求都得排队）。用 iterate_in_threadpool
            # 把它丢到线程池里执行，就不会拖住别的请求。
            async for piece in iterate_in_threadpool(sync_stream):
                yield {
                    "event": "delta",
                    "data": json.dumps({"text": piece}, ensure_ascii=False),
                }

            yield {"event": "done", "data": "{}"}

        except Exception as exc:                              # noqa: BLE001
            yield {
                "event": "error",
                "data": json.dumps({"message": str(exc)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    # 用 python -m app.api 启动，等价于 uvicorn app.api:app
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
