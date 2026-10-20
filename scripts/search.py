"""只做检索、不调用大模型 —— 排查 RAG 问题最常用的工具。

为什么需要它？
    当问答结果不对时，要先判断是"检索没找对资料"还是"模型没答好"。
    这个脚本只看检索结果，能把问题定位到具体一环。

用法（在项目根目录下执行）：
    python scripts/search.py "比亚迪2024年的营业收入是多少"
    python scripts/search.py "招商银行不良贷款率" --k 8
"""

import argparse
import sys
from pathlib import Path

# 本脚本在 scripts/ 下，需要把项目根目录加进搜索路径才能 import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app import config  # noqa: E402  必须最先导入
from app.rag import retrieve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="只检索不生成，用于调试向量检索效果")
    parser.add_argument("question", help="查询语句")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="返回几条")
    args = parser.parse_args()

    print(f"查询：{args.question}")
    print(f"向量库：{config.COLLECTION_NAME}（{config.CHROMA_DIR}）")
    print("=" * 70)

    hits = retrieve(args.question, k=args.k)
    for i, hit in enumerate(hits, start=1):
        print(f"\n[{i}] {hit['company']} 第{hit['page']}页   距离={hit['distance']}")
        # 把换行去掉，方便一眼扫过
        preview = hit["text"].replace("\n", " ")[:160]
        print(f"    {preview}...")


if __name__ == "__main__":
    main()
