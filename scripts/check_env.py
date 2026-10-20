"""环境自检：确认依赖装好、向量模型能下载、中文相似度正常。

用途：
    - 第一次跑项目时确认环境没问题
    - 出问题时先跑它，能快速定位是哪一环坏了

运行：
    python scripts/check_env.py
"""

import sys
import time

# 让 app 包可以被找到（本脚本在 scripts/ 下，需要把项目根目录加进搜索路径）
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app import config  # noqa: E402  必须最先导入：设置镜像和缓存目录

print("=" * 60)
print("环境自检")
print("=" * 60)
print(f"Python      : {sys.version.split()[0]}")
print(f"项目根目录  : {config.ROOT}")
print(f"HuggingFace : {config.EMBEDDING_MODEL}")
print(f"镜像地址    : {config.EMBEDDING_MODEL and __import__('os').environ.get('HF_ENDPOINT')}")
print(f"模型缓存    : {config.MODEL_DIR}")
print()

# ---------- ① 依赖 ----------
print("[1/3] 检查依赖包 ...")
try:
    import chromadb
    import fastapi
    import fastembed
    import langchain_openai
    import pypdf
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"      fastapi={fastapi.__version__}  chromadb={chromadb.__version__}")
    print("      全部导入成功")
except Exception as exc:                                   # noqa: BLE001
    print(f"      失败：{exc}")
    print("      请执行：pip install -r requirements.txt")
    sys.exit(1)

# ---------- ② 向量模型 ----------
print("\n[2/3] 加载中文向量模型（首次运行需下载约 100MB）...")
try:
    from fastembed import TextEmbedding

    t0 = time.time()
    model = TextEmbedding(model_name=config.EMBEDDING_MODEL, cache_dir=str(config.MODEL_DIR))
    print(f"      模型就绪，耗时 {time.time() - t0:.1f} 秒")
except Exception as exc:                                   # noqa: BLE001
    print(f"      失败：{exc}")
    print("      排查方向：网络能否访问 hf-mirror.com；或删掉 data/models 重试")
    sys.exit(1)

# ---------- ③ 相似度是否合理 ----------
print("\n[3/3] 验证向量语义（相关句子的相似度应该明显更高）...")
import numpy as np

sentences = {
    "问题": "比亚迪2024年的营业收入是多少",
    "相关": "公司2024年实现营业收入7771亿元，同比增长29%",
    "无关": "本公司独立董事对关联交易事项发表了同意的意见",
}
vectors = {k: np.array(v) for k, v in zip(sentences, model.embed(list(sentences.values())))}


def cosine(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


q = vectors["问题"]
rel = cosine(q, vectors["相关"])
irr = cosine(q, vectors["无关"])
print(f"      与「相关句子」相似度 = {rel:.4f}")
print(f"      与「无关句子」相似度 = {irr:.4f}")

if rel > irr:
    print("\n结论：环境正常，可以开始建库了。")
    print("下一步：python -m app.ingest")
else:
    print("\n结论：相似度异常（相关句反而不如无关句），请检查模型是否下载完整。")
    sys.exit(1)
