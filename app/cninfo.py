"""巨潮资讯网数据抓取（可复用模块）。

为什么把这段逻辑单独放在 app/ 里，而不是留在 scripts/ 脚本里？
    因为"从巨潮拿数据"这件事有两个使用场景：
        1. 命令行手动抓取      → scripts/fetch_docs.py
        2. 建库时自动抓取      → python -m app.ingest --fetch
    逻辑写在脚本里，app/ingest.py 就 import 不到（scripts 不是包）。
    把可复用的核心逻辑放进 app/，脚本只做"参数解析 + 打印"，这是常见的分层做法。

巨潮资讯网是证监会指定的信息披露网站，A 股所有上市公司公告都免费公开。
"""

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from app import config  # noqa: F401  导入它会设置 HF 镜像等环境变量

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------- 巨潮的两个接口 ----------
# 1) 查询公告列表（POST）
QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
# 2) 公告正文托管在 static 域名下
STATIC_URL = "http://static.cninfo.com.cn/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}

# 公告分类代码（巨潮内部编码）
CATEGORIES = {
    "年报": "category_ndbg_szsh",
    "半年报": "category_bndbg_szsh",
    "一季报": "category_yjdbg_szsh",
    "三季报": "category_sjdbg_szsh",
    "业绩预告": "category_yjyg_szsh",
}

# 默认抓这几家：知名度高、业务好理解，演示和面试时都拿得出手
DEFAULT_COMPANIES = ["贵州茅台", "宁德时代", "比亚迪", "美的集团", "招商银行", "五粮液"]

# 标题里带这些词的不要（摘要 / 英文版 / 更正公告 / 审计报告等都不是正文）
EXCLUDE_WORDS = ["摘要", "英文", "更正", "取消", "补充", "问询", "意见", "审计报告", "公告"]


def strip_em(text: str) -> str:
    """巨潮的搜索结果会用 <em> 标签高亮关键词，这里去掉。"""
    return re.sub(r"</?em>", "", text or "").strip()


def query_announcements(searchkey: str, category_code: str, page_size: int = 20) -> list[dict]:
    """向巨潮查询公告列表，返回公告字典列表。"""
    form = {
        "pageNum": "1",
        "pageSize": str(page_size),
        "column": "szse",              # szse 同时覆盖深市和沪市
        "tabName": "fulltext",
        "plate": "",
        "stock": "",
        "searchkey": searchkey,
        "secid": "",
        "category": category_code,
        "trade": "",
        "seDate": "2024-01-01~2025-12-31",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    body = urllib.parse.urlencode(form).encode("utf-8")
    req = urllib.request.Request(QUERY_URL, data=body, headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("announcements") or []


def pick_main_report(announcements: list[dict]) -> dict | None:
    """从一堆公告里挑出真正的"年度报告"正文，跳过摘要、英文版、更正公告等。"""
    for ann in announcements:
        title = strip_em(ann.get("announcementTitle", ""))
        if not ann.get("adjunctUrl"):
            continue
        if "年度报告" not in title:
            continue
        if any(word in title for word in EXCLUDE_WORDS):
            continue
        return ann
    return None


def download_pdf(adjunct_url: str, dest: Path) -> int:
    """下载一份公告 PDF，返回文件字节数。"""
    url = STATIC_URL + adjunct_url.lstrip("/")
    req = urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    return total


def safe_filename(text: str) -> str:
    """把公司名/标题变成 Windows 合法文件名。"""
    for ch in '\\/:*?"<>|':
        text = text.replace(ch, "_")
    text = re.sub(r"\s+", " ", text).strip().rstrip(".")
    return text[:80]


def fetch_annual_reports(
    companies: list[str],
    category: str = "年报",
    raw_dir: Path | None = None,
    quiet: bool = False,
) -> list[dict]:
    """按公司名下载公告 PDF 到本地，返回成功下载的元信息列表。

    已经存在的文件会跳过下载（这份 PDF 就是本地缓存，避免重复请求巨潮）。
    """
    raw_dir = raw_dir or config.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    category_code = CATEGORIES.get(category, CATEGORIES["年报"])

    def say(msg: str) -> None:
        if not quiet:
            print(msg)

    records: list[dict] = []
    for company in companies:
        say(f"[查询] {company}")
        try:
            announcements = query_announcements(company, category_code)
        except Exception as exc:                              # noqa: BLE001
            say(f"       查询失败：{exc}")
            continue

        target = pick_main_report(announcements)
        if target is None:
            say(f"       没找到合适的{category}正文，跳过")
            continue

        sec_name = strip_em(target.get("secName", company))
        title = strip_em(target.get("announcementTitle", "未命名"))

        # 跳过 ST / *ST 公司：经营异常，数据不适合做演示
        if "ST" in sec_name.upper():
            say(f"       跳过（ST 公司）：{sec_name}")
            continue

        filename = f"{safe_filename(sec_name)}_{safe_filename(title)}.pdf"
        dest = raw_dir / filename

        if dest.exists():
            say(f"       已存在，跳过下载：{filename}")
        else:
            try:
                size = download_pdf(target["adjunctUrl"], dest)
                say(f"       下载完成：{size / 1024:.0f} KB -> {filename}")
            except Exception as exc:                          # noqa: BLE001
                say(f"       下载失败：{exc}")
                continue
            time.sleep(0.5)                                    # 对服务器友好一点

        records.append(
            {
                "company": sec_name,
                "title": title,
                "file": filename,
                "announcement_time": target.get("announcementTime"),
                "url": STATIC_URL + target["adjunctUrl"].lstrip("/"),
            }
        )

    index_file = raw_dir / "index.json"
    index_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records
