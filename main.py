"""项目启动器 —— 在 PyCharm 里右键本文件选 Run，或直接点右上角绿色三角形。

为什么需要这个文件？
    项目里的代码用的是 `from app import config` 这种包导入方式，
    所以必须从项目根目录运行（例如 `python -m app.cli`）。
    如果直接右键 `app/cli.py` 选 Run，Python 会把 `app/` 当成搜索路径，
    找不到 `app` 包，报错 `No module named app`。

    而本文件放在【项目根目录】，Python 运行它时会自动把项目根目录
    加进模块搜索路径，所以 `from app import xxx` 能正常工作 ——
    于是它天生就能一键运行，不需要配任何东西。

用法：
    点绿色三角形        → 弹出菜单，选一个功能
    python main.py cli  → 直接进命令行问答
    python main.py api  → 直接启动后端
    python main.py help → 看所有可用命令
"""

import subprocess
import sys
from pathlib import Path

# 项目根目录（本文件所在目录）
ROOT = Path(__file__).resolve().parent

# Windows 控制台默认不是 UTF-8，中文会乱码2
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BANNER = r"""
==============================================================
   上市公司年报 RAG 智能问答系统
==============================================================
"""

MENU = """
请选择要执行的操作：

  [1] 命令行问答          ← 推荐先试这个，直接用命令行提问
  [2] 启动 Web 后端       FastAPI，端口 8000（Swagger 文档在 /docs）
  [3] 启动 Web 前端       React，端口 3000（需要先装 Node 依赖）
  [4] 环境自检            检查依赖、向量模型、语义相似度是否正常
  [5] 重新建立向量库      把 data/raw 里的 PDF 重新灌进 Chroma
  [6] 从巨潮下载年报      自动抓取上市公司公告 PDF
  [7] 只测检索            不调用大模型，只看检索结果（排查问题用）
  [8] 跑评测              30 题评测集，输出 Recall@k 等指标
  [0] 退出
"""


def _pause() -> None:
    """等用户按回车，避免执行完菜单直接刷新看不到结果。"""
    try:
        input("\n按回车返回菜单...")
    except (KeyboardInterrupt, EOFError):
        pass


def run_cli() -> None:
    """命令行问答。直接在进程内调用，这样 input() 能正常工作。"""
    from app.cli import main as cli_main

    cli_main()


def run_api() -> None:
    """启动 FastAPI 后端（阻塞，Ctrl+C 或点红色方块停止）。"""
    import uvicorn

    print("\n后端启动中... 浏览器打开 http://127.0.0.1:8000/docs 可以测试接口")
    print("停止服务：按 Ctrl+C，或点 PyCharm 的红色停止按钮\n")
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)


def run_web() -> None:
    """启动 React 前端开发服务器（阻塞）。"""
    web_dir = ROOT / "web"
    if not web_dir.exists():
        print("找不到 web/ 目录")
        return
    if not (web_dir / "node_modules").exists():
        print("检测到前端依赖还没安装，先执行 npm install（只需一次，约 1 分钟）...\n")
        subprocess.run("npm install", cwd=web_dir, shell=True, check=False)

    print("\n前端启动中... 启动后浏览器打开 http://localhost:3000")
    print("注意：请用 Vite 打印出来的地址，Windows 上通常只能通过 localhost 访问")
    print("停止服务：按 Ctrl+C，或点 PyCharm 的红色停止按钮\n")
    subprocess.run("npm run dev", cwd=web_dir, shell=True, check=False)


def run_check() -> None:
    """环境自检。"""
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_env.py")], cwd=ROOT, check=False)


def run_ingest() -> None:
    """重建向量库。"""
    subprocess.run([sys.executable, "-m", "app.ingest"], cwd=ROOT, check=False)


def run_fetch() -> None:
    """从巨潮下载年报。"""
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_docs.py")], cwd=ROOT, check=False)


def run_search() -> None:
    """只检索不生成，用于排查检索问题。"""
    question = input("\n请输入要检索的问题：").strip()
    if not question:
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "search.py"), question],
        cwd=ROOT,
        check=False,
    )


def run_eval() -> None:
    """跑评测集。"""
    subprocess.run([sys.executable, str(ROOT / "eval" / "run_eval.py")], cwd=ROOT, check=False)


# 命令名 -> (说明, 函数)
COMMANDS = {
    "cli": ("命令行问答", run_cli),
    "api": ("启动 Web 后端", run_api),
    "web": ("启动 Web 前端", run_web),
    "check": ("环境自检", run_check),
    "ingest": ("重建向量库", run_ingest),
    "fetch": ("从巨潮下载年报", run_fetch),
    "search": ("只测检索", run_search),
    "eval": ("跑评测", run_eval),
}

# 菜单编号 -> 命令名
MENU_MAP = {
    "1": "cli",
    "2": "api",
    "3": "web",
    "4": "check",
    "5": "ingest",
    "6": "fetch",
    "7": "search",
    "8": "eval",
}


def print_help() -> None:
    print(BANNER)
    print("用法：python main.py <命令>\n")
    print("可用命令：")
    for name, (desc, _) in COMMANDS.items():
        print(f"  {name:<8} {desc}")
    print("\n不加任何参数运行，会弹出交互菜单（推荐用这种方式）。")


def interactive() -> None:
    """交互菜单模式。"""
    print(BANNER)
    while True:
        print(MENU)
        try:
            choice = input("请输入编号后回车：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出。")
            return

        if choice in ("0", "q", "quit", "exit"):
            print("已退出。")
            return
        if not choice:
            continue

        command = MENU_MAP.get(choice)
        if command is None:
            print(f"\n没有编号 {choice} 这个选项，请重新输入。")
            continue

        desc, func = COMMANDS[command]
        print(f"\n{'=' * 62}\n  {desc}\n{'=' * 62}\n")
        try:
            func()
        except KeyboardInterrupt:
            print("\n（已中断）")
        except Exception as exc:                              # noqa: BLE001
            print(f"\n[出错] {type(exc).__name__}: {exc}")
            print("提示：如果是依赖或向量库的问题，先选 [4] 环境自检 看看。")

        # 阻塞型功能（服务）退出后不需要再按回车
        if command not in ("api", "web"):
            _pause()


def main() -> None:
    args = sys.argv[1:]

    if not args:
        interactive()
        return

    command = args[0].lower()
    if command in ("help", "-h", "--help"):
        print_help()
        return

    if command not in COMMANDS:
        print(f"未知命令：{command}\n")
        print_help()
        return

    _desc, func = COMMANDS[command]
    func()


if __name__ == "__main__":
    main()
