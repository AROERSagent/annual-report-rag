"""建立向量库：把 data/raw 里的年报 PDF 变成可检索的 Chroma 知识库。

运行（在项目根目录下）：
    python -m app.ingest                # 全量重建
    python -m app.ingest --max-pages 40 # 每份只取前 40 页，快速试跑

流程：PDF → 按页抽文本 → 中文分块 → 向量化 → 存入 Chroma

三个设计决定，都是为了"可理解、可调试"：

1. 为什么按页处理？
   因为答案要给出"来自第几页"的引用。整份文档一起切块会丢失页码。

2. 为什么用 PyMuPDF 而不是 pypdf？
   实测发现 pypdf 对部分中文年报（自定义 CID 字体编码）会抽出乱码，
   加装 fontTools 也修不好；PyMuPDF 和 pdfplumber 都能正常抽取。
   PyMuPDF 是 C 实现，速度快很多，所以选它。
   —— 这是 RAG 里最容易被忽略的一课：抽取质量决定整个系统的上限。

3. 为什么自己算向量，不用 Chroma 内置的 embedding function？
   因为自己算看得见：embed() 出来就是一堆数字，然后显式存进去。
   Chroma 只当"存数字 + 快速找最近邻"的数据库用，职责单一，出错好排查。
"""

import argparse
import re
import sys
import time

import chromadb
import pymupdf
from app import config  # noqa: F401  必须第一个导入：设置 HF 镜像地址
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 中文分块的分隔符优先级：先按段落切，再按句子切，最后才硬切
SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]


def clean_text(text: str) -> str:
    """清掉 PDF 抽取常见的噪音。

    最重要的一步是"修复被空格切碎的中文词"：
        PDF 按字形排版，抽取出来的中文经常在词中间带空格，
        例如「不良 贷款率」「毛 利率」。实测本库里「不良贷款率」
        共出现 27 次，其中 13 次被切碎——这会让关键词检索直接失效。
        所以要把"两个中文字之间的空白"删掉。
        注意只删中文与中文之间的空格，不能碰英文单词和数字。
    """
    text = text.replace("\u3000", " ")          # 全角空格
    text = re.sub(r"[ \t]+", " ", text)          # 连续空格压成一个
    # 关键：删掉中文字符之间的空白（含换行）
    text = re.sub(r"(?<=[\u4e00-\u9fff])[\s]+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)       # 连续空行压成两个
    return text.strip()


def load_pdf(path, splitter, max_pages: int = 0) -> list[dict]:
    """把一份 PDF 读成一批文本块（dict 形式，方便直接喂给 Chroma）。"""
    company = path.stem.split("_")[0]            # 文件名前缀就是公司名
    chunks: list[dict] = []

    with pymupdf.open(str(path)) as doc:
        for page_no, page in enumerate(doc, start=1):
            if max_pages > 0 and page_no > max_pages:
                break

            text = clean_text(page.get_text())
            if len(text) < 30:                    # 太短的页（封面、目录）跳过
                continue

            for piece in splitter.split_text(text):
                chunks.append(
                    {
                        "text": piece,
                        "company": company,
                        "file": path.name,
                        "page": page_no,
                    }
                )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把年报 PDF 灌进 Chroma 向量库",
        epilog="示例：python -m app.ingest --fetch          # 先下载再建库，一步到位",
    )
    parser.add_argument("--max-pages", type=int, default=0, help="每份 PDF 最多取几页，0=全部")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="建库前先从巨潮资讯网下载年报（已存在的 PDF 会跳过，不会重复下载）",
    )
    parser.add_argument(
        "--companies",
        default=None,
        help="配合 --fetch 使用，指定公司名（逗号分隔）。不填则用默认的 6 家知名公司",
    )
    args = parser.parse_args()

    # ---------- ⓪ 可选：先自动下载数据 ----------
    # 注意：PDF 会保存到 data/raw/ 作为本地缓存，而不是直接读进内存。
    # 这样做的好处：重复建库不用重新下载（省时间、也对巨潮服务器友好）、
    # 出问题时可以打开 PDF 核对抽取结果、数据来源可追溯。
    if args.fetch:
        from app.cninfo import DEFAULT_COMPANIES, fetch_annual_reports

        companies = (
            [c.strip() for c in args.companies.split(",") if c.strip()]
            if args.companies
            else DEFAULT_COMPANIES
        )
        print(f"[0/3] 先从巨潮资讯网下载年报（{len(companies)} 家）...\n")
        fetch_annual_reports(companies)
        print()

    if not config.RAW_DIR.exists() or not list(config.RAW_DIR.glob("*.pdf")):
        print("没有找到 PDF。两种办法：")
        print("  python -m app.ingest --fetch    # 自动从巨潮下载后再建库")
        print("  python scripts/fetch_docs.py    # 只下载，稍后再建库")
        return

    pdf_files = sorted(config.RAW_DIR.glob("*.pdf"))
    print(f"发现 {len(pdf_files)} 份 PDF\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    # ---------- ① 抽取 + 分块 ----------
    print("[1/3] 抽取文本并分块（PyMuPDF）...")
    all_chunks: list[dict] = []
    start = time.time()
    for pdf in pdf_files:
        chunks = load_pdf(pdf, splitter, args.max_pages)
        all_chunks.extend(chunks)
        print(f"      {pdf.name[:44]:<46} {len(chunks):>5} 块")

    if not all_chunks:
        print("没有抽到任何文本，请检查 PDF 是否正常")
        return
    print(f"      合计 {len(all_chunks)} 块，抽取耗时 {time.time() - start:.1f} 秒\n")

    # ---------- ② 向量化 ----------
    print(f"[2/3] 加载向量模型 {config.EMBEDDING_MODEL}")
    print("      首次运行会自动下载模型文件（约 100MB，走 hf-mirror 镜像）...")
    model = TextEmbedding(model_name=config.EMBEDDING_MODEL, cache_dir=str(config.MODEL_DIR))

    print(f"      正在向量化 {len(all_chunks)} 块（CPU 上需要几分钟）...")
    start = time.time()
    texts = [c["text"] for c in all_chunks]
    vectors = [v.tolist() for v in tqdm(model.embed(texts), total=len(texts), unit="块")]
    print(f"      完成，耗时 {time.time() - start:.1f} 秒\n")

    # ---------- ③ 存入 Chroma ----------
    print("[3/3] 写入 Chroma 向量库 ...")
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    # 重建前先删掉旧集合，避免新旧数据混在一起。
    #
    # 为什么是删「集合」而不是删整个目录？
    #   1. 更安全：只影响这一个集合。以后项目里如果有多个集合（比如按公司分库），
    #      删目录会把它们全干掉
    #   2. 更省事：不需要文件系统删除操作。如果后端服务正开着把数据文件锁住，
    #      删目录会失败（Windows 会报"文件正由另一进程使用"），
    #      但删集合是数据库层面的操作，不会受这个影响
    try:
        client.delete_collection(name=config.COLLECTION_NAME)
        print(f"      已清空旧集合：{config.COLLECTION_NAME}")
    except Exception:                                          # noqa: BLE001
        pass  # 第一次运行时集合还不存在，属于正常情况

    collection = client.create_collection(name=config.COLLECTION_NAME)

    # Chroma 的 add 一次全塞进去会比较慢，这里每 500 块提交一次
    batch = 500
    for i in tqdm(range(0, len(all_chunks), batch), unit="批"):
        part = all_chunks[i : i + batch]
        collection.add(
            ids=[f"{c['file']}-{c['page']}-{j}" for j, c in enumerate(part, start=i)],
            documents=[c["text"] for c in part],
            metadatas=[
                {"company": c["company"], "file": c["file"], "page": c["page"]}
                for c in part
            ],
            embeddings=vectors[i : i + batch],
        )

    print(f"\n完成！向量库位置：{config.CHROMA_DIR}")
    print(f"集合：{config.COLLECTION_NAME}｜共 {collection.count()} 块")
    print("\n下一步：python scripts/search.py \"你的问题\"   （先看检索对不对）")


if __name__ == "__main__":
    main()
