"""命令行问答入口 —— 阶段 1 的成果，跑通它说明整条 RAG 链路是通的。

运行（在项目根目录下）：
    python -m app.cli
"""

import sys

from app import config  # noqa: F401  必须第一个导入

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BANNER = """
============================================================
  上市公司年报 RAG 问答助手（命令行版）
  输入问题回车即可提问，输入 q 退出
  示例：比亚迪 2024 年的营业收入是多少？
============================================================
"""


def show_sources(sources: list[dict]) -> None:
    """打印答案的参考来源。字段名必须和 rag.to_sources() 返回的一致。"""
    print("\n--- 参考来源 ---")
    for src in sources:
        print(f"  · {src['company']} 第{src['page']}页  (相似度距离 {src['distance']})")


def main() -> None:
    from app.rag import answer          # 延迟导入：避免没建向量库时就报错

    print(BANNER)
    while True:
        # 单独捕获退出信号，让 Ctrl+C 能干净地结束
        try:
            question = input("你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见。")
            break

        if not question:
            continue
        if question.lower() in {"q", "quit", "exit", "退出"}:
            print("再见。")
            break

        # 单个问题出错不应该终止整个会话，而且要打印出异常类型，
        # 否则只会看到一句 "'score'" 这种完全无法定位的消息。
        try:
            result = answer(question)
        except Exception as exc:                              # noqa: BLE001
            print(f"\n[出错] {type(exc).__name__}: {exc}")
            print("（这个问题跳过了，可以继续提问；输入 q 退出）")
            continue

        print(f"\n助手：{result['answer']}")
        show_sources(result["sources"])
        print()


if __name__ == "__main__":
    main()
