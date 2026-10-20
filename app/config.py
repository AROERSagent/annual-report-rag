"""全项目的配置中心。

所有可调参数都集中在这里，其它文件只 import 不重复定义。
配置从项目根目录的 .env 文件读取，改参数不用改代码。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------- 项目根目录 ----------
# __file__ 是 app/config.py，.parent 是 app/，再 .parent 才是项目根
ROOT = Path(__file__).resolve().parent.parent

# 读取 .env（必须在读环境变量之前）
load_dotenv(ROOT / ".env")

# ---------- 关键：模型下载相关环境变量 ----------
# 这几行必须在导入 fastembed / huggingface_hub 之前执行，
# 所以本文件永远是项目里第一个被 import 的模块。
#
# 1) 镜像：huggingface.co 在国内基本连不上，hf-mirror.com 是常用镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------- 目录 ----------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"           # 原始 PDF
CHROMA_DIR = DATA_DIR / "chroma"     # 向量库持久化目录
MODEL_DIR = DATA_DIR / "models"      # 向量模型缓存（约 100MB，首次运行自动下载）
FILTER_DIR = DATA_DIR / "filter"     # 分词过滤缓存

# 2) 缓存位置：默认会写到系统临时目录，那里可能被清理导致每次重新下载。
#    放进项目内，一次下载永久复用，项目也自包含、可整体拷贝。
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FILTER_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(MODEL_DIR))
os.environ.setdefault("HF_HOME", str(MODEL_DIR / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# 关闭 hf-xet 传输后端：它会额外开一个进程和缓存目录，容易在受限环境里出错
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# ---------- 大模型（DeepSeek，接口格式兼容 OpenAI）----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ---------- 向量模型（本地运行，免费）----------
# bge-small-zh-v1.5 是中文效果最好的小模型之一，模型文件约 100MB，首次运行自动下载
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# ---------- 检索与分块参数 ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))       # 每块目标字数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))  # 相邻块重叠字数，防止句子被切断
TOP_K = int(os.getenv("TOP_K", "5"))                   # 每次检索取回几块
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "annual_reports")

# ---------- 检索模式（阶段 5 做消融实验用）----------
# vector        : 只用向量检索（基线）
# hybrid        : 向量 + BM25 关键词检索，用 RRF 融合（推荐默认，延迟几乎不增加）
# hybrid_rerank : 上面基础上再加交叉编码器精排（效果最好，但延迟约 3.4 倍）
#
# 实测（30 题评测集，详见 README 的消融实验表）：
#   vector         Recall@5 = 65%   延迟 1.11s
#   hybrid         Recall@5 = 85%   延迟 1.12s   ← 免费提升
#   hybrid_rerank  Recall@5 = 90%   延迟 3.77s
# 所以默认用 hybrid；对准确率要求更高时再开 hybrid_rerank。
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")

# 每一种检索先各自取出多少个候选，再融合
# 取大一点再融合，比直接取 TOP_K 效果好 —— 给融合留出挑选空间
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))

# RRF 公式里的平滑常数。业界经验值 60，一般不用改
RRF_K = int(os.getenv("RRF_K", "60"))

# ---------- 重排模型 ----------
# 交叉编码器，比向量检索慢很多，所以只对少量候选做精排
# 换更小的模型可以改成 BAAI/bge-reranker-v2-m3-int8（约 0.57GB）
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))  # 送进重排模型的候选数
