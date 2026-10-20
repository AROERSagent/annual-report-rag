"""跑评测，算出一组能写进简历的真实数字。

评测 5 个指标，每个都有明确的定义（面试被追问时能讲清楚）：

1. Recall@k / MRR      —— 检索质量
   标准答案所在的页，有没有出现在检索结果的前 k 条里。

2. 引用有效率          —— 有没有编造引用
   把回答里所有 [公司 第X页] 解析出来，检查这些页是否真的在检索结果里。
   如果答案引用了没检索到的页，说明引用是编的。

3. 忠实度              —— 幻觉程度
   让大模型当评审：回答里的每个结论是否都能在检索到的原文中找到依据。

4. 拒答准确率          —— 幻觉控制的另一面
   对语料里根本没有的问题（比如问特斯拉），模型是否老实说"查不到"。

5. 延迟                —— 工程指标
   单次问答的耗时均值、P50、P95。

用法（在项目根目录下）：
    python eval/run_eval.py                  # 全量评测
    python eval/run_eval.py --limit 5        # 只跑前 5 题（快速试跑）
"""

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app import config  # noqa: E402
from app.rag import answer, format_context, retrieve  # noqa: E402

QUESTIONS_FILE = ROOT / "eval" / "questions.jsonl"
RESULT_DIR = ROOT / "data" / "eval_results"

# 匹配答案里的引用标记，例如 [比亚迪 第273页]
CITE_RE = re.compile(r"\[([^\[\]]+?)\s*第\s*(\d+)\s*页\]")

# 判断模型是否在"拒答"。提示词里要求查不到时回答"年报中没有找到相关说明"，
# 这里放宽一点，覆盖几种常见说法。
REFUSAL_MARKERS = [
    "没有找到", "未找到", "无法确定", "没有相关", "未提及", "没有提及",
    "无法回答", "不包含", "没有提及", "无法判断", "没有提供",
]

# 忠实度评审的提示词
JUDGE_PROMPT = """你是严格的评审员。下面有【参考资料】和【模型回答】。

请判断：模型回答里的每一个结论，是否都能在参考资料中找到明确依据？
- 如果有任何结论在参考资料里找不到依据（包括编造、自行推测、用了资料外的常识），判 0
- 如果所有结论都有依据，判 1

只输出一个字符：0 或 1。不要输出任何其它内容。

【参考资料】
{context}

【模型回答】
{answer}"""


def judge_faithfulness(context: str, ans: str) -> int | None:
    """让大模型判断回答是否忠实于参考资料，返回 1/0，失败返回 None。"""
    from app.rag import get_llm

    try:
        raw = str(get_llm().invoke(JUDGE_PROMPT.format(context=context, answer=ans)).content)
    except Exception:                                         # noqa: BLE001
        return None
    match = re.search(r"[01]", raw)
    return int(match.group()) if match else None


def is_refusal(text: str) -> bool:
    return any(marker in text for marker in REFUSAL_MARKERS)


# 上下文召回率的评审提示词
#
# 为什么需要这个指标？
#   最初我用的是"页级命中"：标准答案是出题时那段原文所在的页，
#   检索结果里有没有这一页。但实测发现这个指标会**大量假阴性**：
#   比如"招商银行不良贷款率"的标准答案标在第 44 页，而检索返回的第 33 页
#   是「3.4.3 按行业划分的贷款及不良贷款分布情况」，里面明明就有不良贷款率。
#   原因是同一信息在年报里会出现多次（摘要、正文、附注各一遍）。
#
#   RAG 真正该问的是："检索到的资料够不够回答问题"，
#   而不是"有没有命中某一个特定页"。所以补了这个大模型判定的指标。
CONTEXT_JUDGE_PROMPT = """你是严格的评审员。下面有【问题】和【检索到的参考资料】。

请判断：参考资料里是否包含回答这个问题所需的信息？

- 只要有任何一段资料包含了回答所需的关键事实，就判 1
- 如果所有资料都不包含，判 0
- 不要自己回答问题，只判断"资料够不够"

只输出一个字符：0 或 1。不要输出任何其它内容。

【问题】
{question}

【参考资料】
{context}"""


def judge_context_recall(question: str, context: str) -> int | None:
    """让大模型判断：检索到的资料够不够回答问题。"""
    from app.rag import get_llm

    try:
        raw = str(get_llm().invoke(
            CONTEXT_JUDGE_PROMPT.format(question=question, context=context)
        ).content)
    except Exception:                                         # noqa: BLE001
        return None
    match = re.search(r"[01]", raw)
    return int(match.group()) if match else None


def evaluate_one(item: dict, top_k: int) -> dict:
    """评测单个问题。"""
    question = item["question"]
    result: dict = {"id": item["id"], "type": item["type"], "question": question}

    # ---------- 检索 ----------
    t0 = time.time()
    hits = retrieve(question, k=top_k)
    result["retrieve_seconds"] = round(time.time() - t0, 3)
    result["retrieved"] = [
        {"company": h["company"], "page": h["page"], "distance": h["distance"]} for h in hits
    ]

    # ---------- Recall@k / MRR（只对"有明确答案页"的题算）----------
    if item.get("gt_page") is not None:
        gt = (item["gt_company"], item["gt_page"])
        rank = None
        for i, h in enumerate(hits, start=1):
            if (h["company"], h["page"]) == gt:
                rank = i
                break
        result["gt_rank"] = rank
        result["recall_at"] = {
            f"@{k}": bool(rank and rank <= k) for k in (1, 3, 5, 10) if k <= top_k
        }
        result["reciprocal_rank"] = round(1 / rank, 4) if rank else 0.0

    # ---------- 生成 ----------
    t0 = time.time()
    ans_result = answer(question, k=top_k)
    result["answer"] = ans_result["answer"]
    result["total_seconds"] = round(time.time() - t0, 3)

    # ---------- 引用有效率 ----------
    retrieved_set = {(h["company"], h["page"]) for h in hits}
    # 公司名可能带"股份有限公司"等后缀，用包含关系做宽松匹配
    cited = CITE_RE.findall(ans_result["answer"])
    if cited:
        valid = 0
        for company, page in cited:
            page = int(page)
            if any(
                page == r_page and (company in r_company or r_company in company)
                for r_company, r_page in retrieved_set
            ):
                valid += 1
        result["citations"] = len(cited)
        result["citation_valid_rate"] = round(valid / len(cited), 3)
    else:
        result["citations"] = 0
        result["citation_valid_rate"] = None

    # ---------- 拒答 / 忠实度 ----------
    result["refused"] = is_refusal(ans_result["answer"])

    if item["type"] == "unanswerable":
        # 查不到的题：拒答就是正确答案
        result["correct"] = result["refused"]
    else:
        # 能查到的题：不该拒答，而且要忠实于资料
        result["correct"] = not result["refused"]
        context = format_context(hits)
        result["faithfulness"] = judge_faithfulness(context, ans_result["answer"])
        # 上下文召回率：检索到的资料够不够回答问题（比"页级命中"更贴近真实效果）
        result["context_recall"] = judge_context_recall(question, context)

    return result


def summarize(results: list[dict]) -> dict:
    """汇总成简历上能用的数字。"""
    summary: dict = {"total": len(results)}

    # ---- 检索指标（只看有标准答案页的题）----
    with_gt = [r for r in results if r.get("gt_rank") is not None or "gt_rank" in r]
    if with_gt:
        summary["retrieval"] = {
            "n": len(with_gt),
            "recall@1": round(sum(r["recall_at"].get("@1", False) for r in with_gt) / len(with_gt), 3),
            "recall@3": round(sum(r["recall_at"].get("@3", False) for r in with_gt) / len(with_gt), 3),
            "recall@5": round(sum(r["recall_at"].get("@5", False) for r in with_gt) / len(with_gt), 3),
            "mrr": round(sum(r["reciprocal_rank"] for r in with_gt) / len(with_gt), 3),
        }

    # ---- 引用有效率 ----
    cited = [r for r in results if r.get("citation_valid_rate") is not None]
    if cited:
        summary["citation"] = {
            "n": len(cited),
            "valid_rate": round(sum(r["citation_valid_rate"] for r in cited) / len(cited), 3),
            "fabricated": sum(1 for r in cited if r["citation_valid_rate"] < 1.0),
        }

    # ---- 忠实度 ----
    faith = [r["faithfulness"] for r in results if r.get("faithfulness") is not None]
    if faith:
        summary["faithfulness"] = {
            "n": len(faith),
            "rate": round(sum(faith) / len(faith), 3),
        }

    # ---- 上下文召回率（更贴近真实效果的检索指标）----
    ctx = [r["context_recall"] for r in results if r.get("context_recall") is not None]
    if ctx:
        summary["context_recall"] = {
            "n": len(ctx),
            "rate": round(sum(ctx) / len(ctx), 3),
        }

    # ---- 拒答准确率（按题型）----
    unans = [r for r in results if r["type"] == "unanswerable"]
    if unans:
        summary["refusal"] = {
            "n": len(unans),
            "accuracy": round(sum(r["correct"] for r in unans) / len(unans), 3),
        }
    ans_q = [r for r in results if r["type"] != "unanswerable"]
    if ans_q:
        summary["answerable"] = {
            "n": len(ans_q),
            "answer_rate": round(sum(r["correct"] for r in ans_q) / len(ans_q), 3),
        }

    # ---- 延迟 ----
    totals = sorted(r["total_seconds"] for r in results)
    if totals:
        summary["latency"] = {
            "mean": round(statistics.mean(totals), 2),
            "p50": round(statistics.median(totals), 2),
            "p95": round(totals[max(0, int(len(totals) * 0.95) - 1)], 2),
            "max": round(max(totals), 2),
        }

    return summary


def print_report(summary: dict, results: list[dict]) -> None:
    print("\n" + "=" * 66)
    print("评测报告")
    print("=" * 66)

    r = summary.get("retrieval")
    if r:
        print(f"\n【检索质量·页级】{r['n']} 题（标准答案页有没有进前 k 名）")
        print(f"  Recall@1 = {r['recall@1']:.1%}    Recall@3 = {r['recall@3']:.1%}    "
              f"Recall@5 = {r['recall@5']:.1%}")
        print(f"  MRR      = {r['mrr']:.3f}")
        print("  注：这个指标偏保守 —— 同一信息在年报里常有多个出处，")
        print("      没命中指定页不代表资料里没有，看下面的上下文召回率更准。")

    ctx = summary.get("context_recall")
    if ctx:
        print(f"\n【上下文召回率】{ctx['n']} 题（大模型判定：资料够不够回答问题）")
        print(f"  召回率 = {ctx['rate']:.1%}   ← 这个指标更贴近真实检索效果")

    c = summary.get("citation")
    if c:
        print(f"\n【引用有效率】{c['n']} 题产生了引用")
        print(f"  引用有效 = {c['valid_rate']:.1%}   （{c['fabricated']} 题存在无效引用）")

    f = summary.get("faithfulness")
    if f:
        print(f"\n【忠实度】{f['n']} 题由大模型评审")
        print(f"  忠实率 = {f['rate']:.1%}")

    if summary.get("refusal"):
        print(f"\n【拒答准确率】{summary['refusal']['n']} 题（语料里查不到的）")
        print(f"  正确拒答 = {summary['refusal']['accuracy']:.1%}")
    if summary.get("answerable"):
        print(f"\n【可答题作答率】{summary['answerable']['n']} 题")
        print(f"  未误拒答 = {summary['answerable']['answer_rate']:.1%}")

    lat = summary.get("latency")
    if lat:
        print(f"\n【延迟】")
        print(f"  均值 {lat['mean']}s   P50 {lat['p50']}s   P95 {lat['p95']}s   最慢 {lat['max']}s")

    # 打印答错的题，方便定位问题
    bad = [x for x in results if x.get("correct") is False]
    if bad:
        print(f"\n【答错的题】{len(bad)} 道")
        for x in bad[:8]:
            print(f"  [{x['type']}] {x['question'][:44]}")
            print(f"        -> {x['answer'][:70]}")

    print("\n" + "=" * 66)


def main() -> None:
    parser = argparse.ArgumentParser(description="跑 RAG 评测")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0=全部")
    parser.add_argument("--top-k", type=int, default=5, help="检索条数")
    args = parser.parse_args()

    if not QUESTIONS_FILE.exists():
        print(f"评测集不存在：{QUESTIONS_FILE}")
        print("请先运行：python eval/build_questions.py")
        return

    questions = [json.loads(line) for line in QUESTIONS_FILE.open(encoding="utf-8") if line.strip()]
    if args.limit:
        questions = questions[: args.limit]

    print(f"开始评测：{len(questions)} 题，TOP_K={args.top_k}\n")

    results: list[dict] = []
    for i, item in enumerate(questions, start=1):
        preview = item["question"][:38]
        print(f"[{i:>2}/{len(questions)}] {preview} ... ", end="", flush=True)
        try:
            r = evaluate_one(item, args.top_k)
        except Exception as exc:                              # noqa: BLE001
            print(f"失败 {type(exc).__name__}: {exc}")
            continue

        # 直观地标出这题检索对了没有
        mark = ""
        if "gt_rank" in r:
            mark = f"命中第{r['gt_rank']}位" if r["gt_rank"] else "未命中"
        print(f"{mark}  {r['total_seconds']}s")
        results.append(r)

    if not results:
        print("没有成功评测任何题目")
        return

    summary = summarize(results)
    print_report(summary, results)

    # ---------- 保存结果 ----------
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULT_DIR / f"eval_{stamp}.json"
    out.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n详细结果已保存：{out}")
    print("（这一组数字就是优化前的基线，阶段 5 要和它对比）")


if __name__ == "__main__":
    main()
