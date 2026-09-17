# -*- coding: utf-8 -*-
"""L5 全流程端到端（可选层）：真实 Ollama 模型（bge-m3 embedding + qwen3 LLM）。

验证非 mock 的完整链路：文档导入 → 语义问答 → Memory remember/recall。
前置：ollama serve + bge-m3（必需）+ qwen3:4b（Memory 层 LLM 提取需要）。

用法（crewai uv workspace 根目录）: uv run python delivery/tests/t5_e2e_ollama.py
可选环境变量：E2E_DOCS_DIR（默认 demo-docs 目录；也可用内置样例文本）
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("GAUSSDB_TEST", "1")
os.environ.setdefault("PYTHONUTF8", "1")

OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBED_MODEL = "bge-m3"
DOCS_DIR = os.environ.get("E2E_DOCS_DIR", r"D:\workplace\doc\ObsidianNote\learning\rag\demo-docs")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def ollama_embed(texts: list[str]) -> list[list[float]]:
    req = urllib.request.Request(
        f"{OLLAMA}/api/embed",
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["embeddings"]


def main() -> int:
    import crewai.rag as rag
    from crewai.rag.gaussdb.config import GaussDBRagConfig

    db = os.environ.get("GAUSSDB_DATABASE", "crewai_test")
    rag.config = GaussDBRagConfig(
        embedding_function=ollama_embed,
        database=db,
        user=os.environ.get("GAUSSDB_USER", ""),
        password=os.environ.get("GAUSSDB_PASSWORD", ""),
    )

    # ---------- 1. Knowledge 语义检索 ----------
    from crewai.knowledge.knowledge import Knowledge
    from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

    kn = Knowledge(
        collection_name="delivery_e2e",
        sources=[
            StringKnowledgeSource(content="GaussDB 是华为的企业级分布式数据库，支持 Oracle 兼容模式。"),
            StringKnowledgeSource(content="crewAI 是一个多智能体编排框架，用 Python 编写。"),
        ],
    )
    kn.add_sources()
    hits = kn.query(["华为的企业级数据库是什么"], results_limit=3, score_threshold=0.3)
    top_text = (hits[0]["content"] or "") if hits else ""
    check("Knowledge 语义检索", bool(hits) and ("GaussDB" in top_text or "华为" in top_text),
          f"score={hits[0]['score']:.3f}" if hits else "no hits")

    # ---------- 2. 真实文档导入（可选：E2E_DOCS_DIR 存在时）----------
    from pathlib import Path

    from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

    if os.path.isdir(DOCS_DIR):
        file_paths = [Path(DOCS_DIR) / f for f in sorted(os.listdir(DOCS_DIR)) if f.endswith(".md")]
        kn2 = Knowledge(
            collection_name="delivery_e2e_docs",
            sources=[TextFileKnowledgeSource(file_paths=file_paths)],
        )
        kn2.add_sources()
        hits = kn2.query(["慢 SQL 应该怎么诊断"], results_limit=3, score_threshold=0.3)
        top = hits[0]["content"] if hits else ""
        check("真实文档导入+语义问答", bool(top) and any(
            k in top for k in ("慢 SQL", "执行计划", "诊断")), f"top1 len={len(top)}")
        kn.reset()
    else:
        print(f"[SKIP] 真实文档目录不存在: {DOCS_DIR}")

    # ---------- 3. Memory remember/recall（qwen3 LLM 提取）----------
    try:
        from crewai.memory.unified_memory import Memory

        mem = Memory(storage="gaussdb", embedder=ollama_embed, llm=f"ollama/qwen3:4b")
        mem.remember("用户偏好使用中文交流，项目的数据库选型是华为 GaussDB。")
        import time

        for _ in range(30):
            time.sleep(1)
            if mem._storage.count() > 0:
                break
        count = mem._storage.count()
        hits = mem.recall("项目用什么数据库")
        top_content = hits[0].record.content if hits else ""
        check("Memory remember/recall", count > 0 and ("GaussDB" in top_content or "数据库" in top_content),
              f"count={count}")
        mem.reset()
    except Exception as e:  # noqa: BLE001
        check("Memory remember/recall", False,
              f"{type(e).__name__}: {str(e).splitlines()[0][:100]}（qwen3:4b 未就绪？）")

    try:
        kn.reset()
    except Exception:  # noqa: BLE001
        pass

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n=== L5 {'PASS' if not failed else 'FAIL'}（{len(RESULTS) - len(failed)}/{len(RESULTS)}）===")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
