# Enterprise RAG Knowledge Base

一个面向求职展示的企业知识库智能问答系统 MVP。支持文档上传、解析、切分、向量检索、混合检索、引用溯源、流式问答和会话记录。

## 技术栈

- 后端: FastAPI + SQLAlchemy
- 检索: 自研轻量向量检索 + 关键词混合排序
- 大模型: OpenAI 兼容接口，默认支持 DeepSeek / Qwen / GLM / OpenAI
- 前端: Streamlit
- 存储: SQLite
- 部署: Docker Compose

## 快速启动

环境要求: Python 3.10+。

```powershell
cd E:\ai-rag-kb
.\setup.ps1
.\run_backend.ps1
```

另开一个终端:

```powershell
cd E:\ai-rag-kb
.\run_frontend.ps1
```

打开 http://localhost:8501 使用前端。后端接口文档在 http://localhost:8000/docs。

## 配置大模型

复制 `.env.example` 为 `.env`，按需填写:

```env
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=你的key
LLM_MODEL=deepseek-chat
```

不填写 `LLM_API_KEY` 时，系统自动使用离线 Mock 模式，仍然可以演示完整检索和引用流程。接入真实模型后，回答会由大模型生成。

## 功能

- 上传 PDF、Word、TXT、Markdown 文档
- 文档解析与智能切分
- 向量检索 + 关键词混合召回
- 回答附带文档名、页码和片段引用
- SSE 流式输出
- 会话历史与反馈记录
- 文档列表和知识库统计
- Docker Compose 部署

## API

- `GET /api/health` 健康检查
- `POST /api/documents/upload` 上传文档
- `GET /api/documents` 文档列表
- `DELETE /api/documents/{id}` 删除文档
- `POST /api/chat` 普通问答
- `POST /api/chat/stream` 流式问答
- `GET /api/conversations` 会话列表
- `GET /api/conversations/{id}` 会话详情
- `POST /api/feedback` 提交反馈

## 项目结构

```text
backend/
  app/
    main.py       FastAPI 入口和接口
    config.py     配置管理
    db.py         数据库模型
    rag.py        文档解析、切分、检索、LLM 调用
frontend/
  app.py          Streamlit 前端
samples/          示例文档
scripts/          初始化脚本
tests/            单元测试
data/             运行时产生的数据库和上传文件
```

## 面试讲解重点

1. 为什么使用 RAG: 让回答基于企业文档，降低幻觉，文档更新后无需重新训练模型。
2. 检索流程: 文档切分后生成向量，查询时先用向量召回，再和关键词分数融合排序。
3. 引用溯源: 每个 chunk 保存文档、页码和序号，回答时返回引用片段。
4. 工程化: 配置与代码分离，支持离线 Mock 模式，Docker 一键部署。
5. 可扩展方向: 替换为 Chroma/Milvus、加入 Rerank、Query Rewrite、权限过滤和评测集。

## 注意事项

- `.env` 包含 API Key，不要提交到 GitHub。
- 默认使用哈希向量，优点是零依赖、离线可跑；生产环境建议切换 OpenAI Embedding 或 BGE-M3。
- 当前向量检索适合中小规模知识库。数据量增大后建议切换 Chroma、Milvus 或 pgvector。
## 常用命令

```powershell
# 安装依赖
.\setup.ps1

# 启动后端
.\run_backend.ps1

# 启动前端
.\run_frontend.ps1

# 运行测试
.\.venv\Scripts\python.exe -m pytest -q

# 导入示例文档（需要后端已启动）
.\.venv\Scripts\python.exe scripts\seed_demo.py

# 运行简版评测（需要后端已启动并导入示例文档）
.\.venv\Scripts\python.exe scripts\evaluate.py
```

## 当前实现说明

本项目默认使用 SQLite 保存文档、Chunk、会话和反馈；向量以 JSON 形式保存在 Chunk 表中，查询时使用 NumPy 计算余弦相似度，并融合关键词分数。这个实现适合个人项目和中小规模知识库演示。

如果数据规模变大，可以把 `backend/app/rag.py` 中的检索逻辑替换为 Chroma、Milvus 或 pgvector，接口层不需要大改。