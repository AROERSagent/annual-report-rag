"""从巨潮资讯网下载真实的上市公司公告 PDF。

这个脚本只做三件事：解析命令行参数、调用 app.cninfo 的核心逻辑、打印结果。
真正干活的代码在 app/cninfo.py 里，这样 `python -m app.ingest --fetch`
也能复用同一套逻辑。

用法（在项目根目录下执行）：
    python scripts/fetch_docs.py                      # 默认抓 6 家知名公司的年报
    python scripts/fetch_docs.py --companies 贵州茅台,比亚迪
    python scripts/fetch_docs.py --category 半年报 --companies 宁德时代
    python scripts/fetch_docs.py --list               # 只列出可下载的年报，不下载

下载的文件放在 data/raw/，同时生成 data/raw/index.json 记录元信息。

提示：如果你想让"下载 + 建库"一步完成，直接用：
    python -m app.ingest --fetch
"""

import argparse
import sys
from pathlib import Path

# 本脚本在 scripts/ 下，需要把项目根目录加进搜索路径才能 import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402  必须最先导入
from app.cninfo import CATEGORIES, DEFAULT_COMPANIES, fetch_annual_reports  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="从巨潮资讯网下载上市公司公告 PDF")
    parser.add_argument("--category", default="年报", choices=list(CATEGORIES), help="公告类型")
    parser.add_argument(
        "--companies",
        default=",".join(DEFAULT_COMPANIES),
        help="公司名，逗号分隔（默认 6 家知名公司）",
    )
    parser.add_argument("--list", action="store_true", help="只列出找到的年报，不下载")
    args = parser.parse_args()

    companies = [c.strip() for c in args.companies.split(",") if c.strip()]
    print(f"类型：{args.category}｜公司：{len(companies)} 家\n")

    if args.list:
        # 只查询不下载，方便先看看有哪些能抓
        from app.cninfo import pick_main_report, query_announcements, strip_em

        for company in companies:
            try:
                anns = query_announcements(company, CATEGORIES[args.category])
            except Exception as exc:                          # noqa: BLE001
                print(f"[查询] {company} 失败：{exc}")
                continue
            target = pick_main_report(anns)
            if target:
                print(
                    f"  {strip_em(target.get('secName', company))} - "
                    f"{strip_em(target.get('announcementTitle', ''))}"
                )
            else:
                print(f"  {company} - 没找到合适的{args.category}正文")
        return

    records = fetch_annual_reports(companies, category=args.category)
    print(f"\n完成：共 {len(records)} 份 PDF 在 {config.RAW_DIR}")
    print(f"元信息：{config.RAW_DIR / 'index.json'}")
    print("\n下一步（二选一）：")
    print("  python -m app.ingest            # 用已下载的 PDF 建库")
    print("  python -m app.ingest --fetch    # 下次一步到位：先下载再建库")


if __name__ == "__main__":
    main()
