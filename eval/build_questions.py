"""自动生成评测集（30 题），并记录标准答案。

为什么要自动生成？
    评测 RAG 最难的一步是"标准答案从哪来"。人工翻 1300 页 PDF 标注
    "这个问题答案在第几页"，既慢又容易标错。
    换个思路：从向量库里抽一段原文，让大模型"根据这段原文出一道题"。
    这样原文所在的（公司, 页码）就是标准答案 —— 构造时就确定，100% 准确。

要诚实说明的局限：
    这样生成的问题会偏向原文的措辞，比真人提问更容易被检索到。
    所以另外手工补充了两类题：
      · 数值型（考检索精度）
      · 查不到型（考拒答能力 / 幻觉控制）
    评测结果里会按题型分开统计，不会拿"容易的题"冒充整体水平。

用法（在项目根目录下）：
    python eval/build_questions.py                # 生成 20 道 + 10 道手工题
    python eval/build_questions.py --num 30       # 生成 30 道
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.rag import get_collection, get_llm  # noqa: E402

OUT_FILE = ROOT / "eval" / "questions.jsonl"

# 出题用的提示词。要求"像真人提问"，避免出现"根据以下材料"这种痕迹。
GEN_PROMPT = """下面是一段 A 股上市公司年报的原文。请根据这段内容，出一道具体的、有明确答案的问题。

要求：
1. 要像真实投资者会问的问题，不要出现"根据以下材料""这段文字"之类的说法
2. 问题里必须带上公司名，否则不知道在问谁
3. 只能问这段原文里能直接回答的内容，不要引申
4. 只输出问题本身，不要编号、引号和任何解释

公司名：{company}

年报原文：
{chunk}"""


# ---------------- 手工补充的题目 ----------------
# 数值型：考察检索能不能精确定位到含数字的表格
NUMERIC_QUESTIONS = [
    {"question": "比亚迪 2024 年的营业收入是多少？", "company": "比亚迪"},
    {"question": "贵州茅台的毛利率是多少？", "company": "贵州茅台"},
    {"question": "宁德时代 2024 年的营业收入是多少？", "company": "宁德时代"},
    {"question": "美的集团 2024 年的营业收入是多少？", "company": "美的集团"},
    {"question": "招商银行的不良贷款率是多少？", "company": "招商银行"},
]

# 查不到型：语料里根本没有，考察模型会不会编造
UNANSWERABLE_QUESTIONS = [
    "特斯拉 2024 年的营业收入是多少？",
    "比亚迪明年的股价会涨到多少？",
    "贵州茅台董事长的手机号码是多少？",
    "苹果公司 2024 年卖了多少台 iPhone？",
    "招商银行在火星设有分行吗？",
]


def is_good_chunk(text: str) -> bool:
    """过滤掉不适合出题的文本块。"""
    if not (200 <= len(text) <= 900):
        return False
    # 数字和符号占比过高的块（纯表格）不适合出自然语言问题
    digit_ratio = sum(c.isdigit() or c in ".,%()（）-— " for c in text) / len(text)
    return digit_ratio < 0.55


def sample_chunks(n: int, seed: int = 42) -> list[dict]:
    """从向量库里抽样，尽量覆盖不同公司，保证题目有多样性。"""
    collection = get_collection()
    data = collection.get(include=["documents", "metadatas"])
    docs, metas = data["documents"], data["metadatas"]

    by_company: dict[str, list[dict]] = {}
    for doc, meta in zip(docs, metas):
        if not is_good_chunk(doc):
            continue
        by_company.setdefault(meta["company"], []).append(
            {"text": doc, "company": meta["company"], "page": meta["page"]}
        )

    rng = random.Random(seed)
    # 轮询各公司，避免题目全来自同一家
    for items in by_company.values():
        rng.shuffle(items)
    picked: list[dict] = []
    companies = sorted(by_company)
    idx = 0
    while len(picked) < n and any(by_company[c] for c in companies):
        company = companies[idx % len(companies)]
        if by_company[company]:
            picked.append(by_company[company].pop())
        idx += 1
    return picked


def generate_question(chunk: dict) -> str | None:
    """让大模型根据一段原文出一道题。"""
    llm = get_llm()
    prompt = GEN_PROMPT.format(company=chunk["company"], chunk=chunk["text"][:800])
    try:
        raw = llm.invoke(prompt).content
    except Exception as exc:                                  # noqa: BLE001
        print(f"      生成失败：{exc}")
        return None

    # 清理：去掉引号、编号等模型可能多输出的东西
    question = str(raw).strip().strip('"').strip("「」")
    question = re.sub(r"^\s*\d+[.、)]\s*", "", question).split("\n")[0].strip()
    return question or None


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 RAG 评测集")
    parser.add_argument("--num", type=int, default=20, help="自动生成的题目数量")
    args = parser.parse_args()

    print(f"[1/2] 抽取 {args.num} 个文本块用于出题 ...")
    chunks = sample_chunks(args.num)
    print(f"      抽样完成，覆盖 {len({c['company'] for c in chunks})} 家公司\n")

    print("[2/2] 让大模型逐块出题 ...")
    questions: list[dict] = []

    for i, chunk in enumerate(chunks, start=1):
        question = generate_question(chunk)
        if not question:
            continue
        questions.append(
            {
                "id": f"gen-{len(questions) + 1:03d}",
                "type": "answerable",
                "question": question,
                # 标准答案：这道题就是照着这段原文出的，所以答案必然在这一页
                "gt_company": chunk["company"],
                "gt_page": chunk["page"],
                "source_chunk": chunk["text"][:300],
            }
        )
        print(f"      [{i}/{len(chunks)}] {chunk['company']} 第{chunk['page']}页 -> {question}")

    # 补充数值型题目（标准答案页留空，只评测"能不能检索到对应公司的相关页"）
    for i, item in enumerate(NUMERIC_QUESTIONS, start=1):
        questions.append(
            {
                "id": f"num-{i:03d}",
                "type": "numeric",
                "question": item["question"],
                "gt_company": item["company"],
                "gt_page": None,       # 数值题不指定具体页，只看检索命中的公司对不对
            }
        )

    # 补充"查不到"题目（没有标准答案，考察拒答）
    for i, q in enumerate(UNANSWERABLE_QUESTIONS, start=1):
        questions.append({"id": f"neg-{i:03d}", "type": "unanswerable", "question": q})

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    n_by_type: dict[str, int] = {}
    for q in questions:
        n_by_type[q["type"]] = n_by_type.get(q["type"], 0) + 1

    print(f"\n完成！共 {len(questions)} 题写入 {OUT_FILE}")
    for t, c in sorted(n_by_type.items()):
        print(f"      {t:<14} {c} 题")
    print("\n下一步：python eval/run_eval.py    （跑评测）")


if __name__ == "__main__":
    main()
