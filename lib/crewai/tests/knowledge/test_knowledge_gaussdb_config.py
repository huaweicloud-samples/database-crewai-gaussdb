"""KnowledgeStorage embedder wiring under a global GaussDB RAG config.

``KnowledgeStorage._init_client`` used to hardcode a ``ChromaDBConfig``
whenever an ``embedder`` was passed, so a globally configured
``GaussDBRagConfig`` was ignored (Task 4). These tests pin the new dispatch
(global gaussdb config -> GaussDBClient with the embedder wired in) and the
guardrail (default/unset global config -> unchanged chromadb behavior).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.rag.chromadb.client import ChromaDBClient
from crewai.rag.config import utils as rag_utils
from crewai.rag.gaussdb.client import GaussDBClient
from crewai.rag.gaussdb.config import GaussDBRagConfig

_EMBEDDER: dict[str, Any] = {
    "provider": "openai",
    "config": {"model": "text-embedding-3-small"},
}


@pytest.fixture(autouse=True)
def _isolate_rag_context():
    """Run each test with the global RAG context unset; restore afterwards."""
    previous = rag_utils._rag_context.get()
    rag_utils._rag_context.set(None)
    yield
    rag_utils._rag_context.set(previous)


@pytest.fixture(autouse=True)
def _openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # chromadb's OpenAIEmbeddingFunction refuses to build without a key.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")


def _point_chroma_storage_at_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate the default chromadb persist directory.

    ``DEFAULT_STORAGE_PATH`` is baked at import time, so the CREWAI_STORAGE_DIR
    env-var route cannot isolate this; patch the config-module global that
    ``_default_settings()`` reads at call time instead.
    """
    monkeypatch.setattr(
        "crewai.rag.chromadb.config.DEFAULT_STORAGE_PATH",
        str(tmp_path / "chroma"),
    )


def test_embedder_with_global_gaussdb_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """全局 RAG 配置为 GaussDBRagConfig 时，带 embedder 的 KnowledgeStorage
    应产出 GaussDBClient（而非硬编码的 ChromaDBClient）。"""
    _point_chroma_storage_at_tmp(monkeypatch, tmp_path)

    fake_client = MagicMock(spec=GaussDBClient)
    received: list[GaussDBRagConfig] = []

    def fake_factory(config: GaussDBRagConfig) -> GaussDBClient:
        received.append(config)
        return fake_client

    # 打桩 gaussdb 工厂产出（避免真连库）：注册表先于内建
    # crewai.rag.gaussdb.factory 分支被查询。
    monkeypatch.setattr("crewai.rag.factory._factories", {"gaussdb": fake_factory})

    # 设全局 RAG 配置为 gaussdb（与 `crewai.rag.config = ...` 同一入口）。
    rag_utils.set_rag_config(GaussDBRagConfig(database="unit_test_db"))

    ks = KnowledgeStorage(embedder=_EMBEDDER)

    assert isinstance(ks._client, GaussDBClient)
    assert ks._client is fake_client
    # create_client 运行两次：set_rag_config 建全局 client（无 embedder），
    # KnowledgeStorage 实例 client（embedder 已注入）。
    assert len(received) == 2
    assert received[0].embedding_function is None
    replaced = received[1]
    assert replaced.provider == "gaussdb"
    assert replaced.database == "unit_test_db"  # 全局字段保留
    assert replaced.embedding_function is not None


def test_embedder_default_chromadb_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """未设置全局配置（默认 chromadb）时行为与改造前一致——护栏测试。"""
    _point_chroma_storage_at_tmp(monkeypatch, tmp_path)

    ks = KnowledgeStorage(embedder=_EMBEDDER)

    assert isinstance(ks._client, ChromaDBClient)
