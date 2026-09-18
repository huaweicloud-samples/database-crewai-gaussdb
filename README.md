<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="420px" alt="crewAI — Multi-AI Agent orchestration framework">
  </a>
</p>

<h1 align="center">crewAI on GaussDB</h1>

<p align="center">
  <a href="#-功能总览"><img src="https://img.shields.io/badge/%E9%80%82%E9%85%8D-%E6%8C%81%E4%B9%85%E5%8C%96%20%2B%20RAG%20%E5%90%91%E9%87%8F-blue" alt="适配范围"></a>
  <a href="#-前置条件"><img src="https://img.shields.io/badge/GaussDB-Kernel%20507-orange" alt="GaussDB 507"></a>
  <a href="#-前置条件"><img src="https://img.shields.io/badge/%E9%83%A8%E7%BD%B2-%E9%9B%86%E4%B8%AD%E5%BC%8F%20%2B%20%E5%88%86%E5%B8%83%E5%BC%8F-success" alt="双拓扑"></a>
  <img src="https://img.shields.io/badge/crewAI-1.15.21-green" alt="crewAI 1.15.21">
</p>

本仓库是 [crewAI](https://github.com/crewAIInc/crewAI)（开源多智能体编排框架，[官方文档](https://docs.crewai.com)）v1.15.21 的 **华为 GaussDB 全存储适配版**：crewAI 的全部持久化存储与 RAG 向量存储均可运行在 GaussDB（Oracle 兼容模式）上，替代默认的本地 SQLite / LanceDB / ChromaDB 文件存储，**代码零改动、环境变量一键切换**。

> crewAI 框架本身的功能、API 与文档请参考官方仓库与文档；本 README 只覆盖 GaussDB 适配部分。

## ✨ 功能总览

| crewAI 存储域 | 默认后端 | GaussDB 后端 | 启用方式 |
|---|---|---|---|
| Flow 状态持久化 | SQLite 文件 | 表 `flow_states` / `pending_feedback` | `CREWAI_STORAGE_BACKEND=gaussdb` |
| Kickoff 任务输出（replay/审计） | SQLite 文件 | 表 `latest_kickoff_task_outputs` | 同上 |
| 运行时 Checkpoint | SQLite/JSON 文件 | 表 `checkpoints` | `CheckpointConfig(location="gaussdb", provider=GaussDBProvider())` |
| CLI 读取（checkpoint / log-tasks-outputs） | 读本地文件 | 同库直读 | 跟随同一开关 |
| Memory 统一记忆 | LanceDB 本地 | 表 `memories`（floatvector + GsIVFFLAT/GsDiskANN） | `Memory(storage="gaussdb")` |
| Knowledge / RAG | chromadb 本地 | 表 `crewai_rag_<collection>`（JSONB + floatvector） | `crewai.rag.config = GaussDBRagConfig(...)` |

适配原则：**纯新增后端实现 + 官方扩展点接入**（工厂注册 / Protocol 实现），默认路径零改动；不安装驱动 extra 时行为与上游完全一致。

## 📋 前置条件

| 条件 | 要求 |
|---|---|
| 数据库 | GaussDB Kernel 507+，Oracle 兼容模式（集中式 `DBCOMPATIBILITY='A'` / 分布式 `'ORA'`） |
| 向量功能 | `enable_vectordb=on`（POSTMASTER 级，开启后重启集群） |
| Python | 3.10 ~ 3.13 |
| Embedding 模型 | 集中式 ≤4096 维（≤1024 GsIVFFLAT / 以上 GsDiskANN+PQ 自动选择）；**分布式硬限 1024 维** |
| 认证 | 开源 psycopg2 可连（实例默认 sha256 需按 [delivery/注意事项.md](delivery/注意事项.md) §1 调整为 md5 双存储） |

## 🚀 快速开始

### 1. 安装

```bash
pip install "crewai[gaussdb]"     # = psycopg2-binary>=2.9.12，无其他新依赖
# 源码部署：uv sync --extra gaussdb
```

### 2. 配置

```bash
cp delivery/.env.gaussdb.example .env    # 填写 GAUSSDB_* 连接信息
```

核心变量：`CREWAI_STORAGE_BACKEND=gaussdb`（持久化切换）+ `GAUSSDB_HOST/PORT/USER/PASSWORD/DATABASE`（三个存储域共用）。

### 3. 使用（示例：Crew + Knowledge + Memory 全走 GaussDB）

```python
import crewai.rag as rag
from crewai.rag.gaussdb.config import GaussDBRagConfig
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from pathlib import Path

# Knowledge/RAG → GaussDB（embedding 用任意 Callable，例如 Ollama）
rag.config = GaussDBRagConfig(embedding_function=my_embedder)

crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,                      # Memory → GaussDB（storage="gaussdb" 自动挂载）
    knowledge_sources=[TextFileKnowledgeSource(file_paths=[Path("doc.md")])],
)
result = crew.kickoff()
```

关系持久化（Flow 状态 / Kickoff 输出 / Checkpoint）无需改代码——`CREWAI_STORAGE_BACKEND=gaussdb` 后自动落 GaussDB。

## 🧪 验证部署

```bash
uv run python delivery/tests/run_all.py                  # L1 连通 → L2 SQL → L3 持久化 → L4 向量
uv run python delivery/tests/run_all.py --with-ollama    # + L5 Ollama 真实模型端到端
```

五层测试金字塔的预期结果与失败排查见 [delivery/测试指南.md](delivery/测试指南.md)。

## 🏗️ 适配架构

```
crewAI 应用代码
     │
     ├─ 关系持久化 ── FlowPersistence(ABC) ──┐
     ├─ Kickoff 输出 ── 同签名类 ────────────┤   CREWAI_STORAGE_BACKEND=gaussdb
     ├─ Checkpoint ── BaseProvider(ABC) ─────┤   → GaussDBFlowPersistence /
     │                                        │     GaussDBKickoffTaskOutputsStorage /
     ├─ Memory ── StorageBackend(Protocol) ──┤     GaussDBProvider
     │                                        │   → GaussDBStorage
     └─ Knowledge/RAG ── BaseClient ─────────┘   → GaussDBClient
                                              │
                    crewai.gaussdb（共享层：连接池 / 配置 / 维度门禁 /
                    索引选择 / MERGE+jsonb 批量 upsert / 密码安全）
                                              │
                                     psycopg2（crewai[gaussdb] extra）
                                              │
                              GaussDB 507（O 模式，集中式/分布式）
```

关键设计：
- **O 模式方言适配**：upsert 全走 `MERGE INTO`（O 模式无 `ON CONFLICT`）、自增用 `SEQUENCE`（分布式兼容）、空串→NULL 防御、BOOLEAN/JSONB cast 边界处理——全部经双实例实测
- **向量**：`floatvector` + GsIVFFLAT/GsDiskANN+PQ 自动选择（`pq_nseg` 按维度派生）；Memory score 与 Qdrant 后端对齐，Knowledge 的 doc_id/score 公式精确复刻 chromadb（默认阈值行为不变）
- **安全**：数据库密码全链路排除序列化（永不进入 checkpoint payload / repr）
- **维度门禁**：集中式 ≤4096 / 分布式 ≤1024，启动时校验并给出可操作报错

## 📦 交付文档

| 文档 | 内容 |
|---|---|
| [delivery/实施部署交付文档.md](delivery/实施部署交付文档.md) | 前置条件 / 安装 / 配置 / 部署验证 / 运维 / 已知限制 |
| [delivery/使用文档.md](delivery/使用文档.md) | 各存储域详细用法 + Ollama 示例 |
| [delivery/测试指南.md](delivery/测试指南.md) | 五层测试金字塔 + 基线说明 + 失败排查 |
| [delivery/注意事项.md](delivery/注意事项.md) | 认证配置 / O 模式方言约束表 / 并发锁 / 安全 |

## 🧪 测试

- 交付验证：`delivery/tests/run_all.py`（五层，见上）
- 开发回归：`pytest -n 0`（GaussDB 集成测试以 `GAUSSDB_TEST=1` + `GAUSSDB_*` 门控，~120 个用例双实例全绿；既有 crewAI 测试基线不受影响）

## ⚠️ 已知限制

1. 分布式实例 embedding 维度硬限 1024（CREATE TABLE 级）；高维模型请用集中式或降维
2. `GaussDBClient` 的 metadata 过滤仅支持 dict 等值（chromadb 的 where/where_document 高级语法不支持）
3. CLI checkpoint 的 `diff` 子命令未接 GaussDB 分支（list/info/resume/prune 已支持）
4. 详细清单见 [delivery/注意事项.md](delivery/注意事项.md) §8 与 [delivery/实施部署交付文档.md](delivery/实施部署交付文档.md) §六

## 🔗 相关项目

- [database-n8n-gaussdb](https://github.com/huaweicloud-samples/database-n8n-gaussdb) — n8n 的 GaussDB 适配（同系列 sample）
- [crewAI 官方仓库](https://github.com/crewAIInc/crewAI) · [crewAI 官方文档](https://docs.crewai.com)
- [GaussDB 产品文档](https://support.huaweicloud.com/gaussdb/index.html)

## 📄 License

与上游 [crewAI](https://github.com/crewAIInc/crewAI) 一致（MIT）。
