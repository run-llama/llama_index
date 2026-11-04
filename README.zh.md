# 🗂️ LlamaIndex 🦙

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

[![PyPI - 下载量](https://img.shields.io/pypi/dm/llama-index)](https://pypi.org/project/llama-index/)
[![构建状态](https://github.com/run-llama/llama_index/actions/workflows/build_package.yml/badge.svg)](https://github.com/run-llama/llama_index/actions/workflows/build_package.yml)
[![GitHub 贡献者](https://img.shields.io/github/contributors/jerryjliu/llama_index)](https://github.com/jerryjliu/llama_index/graphs/contributors)
[![Discord 社区](https://img.shields.io/discord/1059199217496772688)](https://discord.gg/dGcwcsnxhU)
[![X / Twitter](https://img.shields.io/twitter/follow/llama_index)](https://x.com/llama_index)
[![Reddit 论坛](https://img.shields.io/reddit/subreddit-subscribers/LlamaIndex?style=plastic&logo=reddit&label=r%2FLlamaIndex&labelColor=white)](https://www.reddit.com/r/LlamaIndex/)

**LlamaIndex OSS**（由 [LlamaIndex](https://llamaindex.ai?utm_medium=li_github&utm_source=github&utm_campaign=2026--) 开发）是用于构建 AI Agent（智能体）与检索增强生成（RAG）应用程序的开源数据框架。**[Parse](https://cloud.llamaindex.ai?utm_medium=li_github&utm_source=github&utm_campaign=2026--)** 是我们专为智能体 OCR、文档解析、结构化提取与索引构建的企业级云平台。你可以将 LlamaParse 与本开源框架配合使用，也可以独立调用。

> ### 📚 **官方技术文档：**
>
> - [LlamaParse 文档](https://developers.llamaindex.ai/python/cloud/llamaparse/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)
> - [LlamaIndex OSS 开源框架文档](https://developers.llamaindex.ai/python/framework/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)
> - [LlamaAgents 智能体框架文档](https://developers.llamaindex.ai/python/llamaagents/overview/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)

使用 LlamaIndex 进行开发通常涉及核心库（Core）以及所选定的生态集成插件（Integrations）。在 Python 中主要有两种起步方式：

1. **开箱即用套件**：[`llama-index`](https://pypi.org/project/llama-index/)。预装核心库及常用精选集成的 Python 启动包。
2. **轻量自定义按需安装**：[`llama-index-core`](https://pypi.org/project/llama-index-core/)。仅安装 LlamaIndex 核心包，并在 [LlamaHub](https://llamahub.ai/) 上按需挑选并安装应用所需的集成插件。LlamaHub 拥有超过 300+ 个无缝兼容核心库的集成包，支持自由搭配任意 LLM、Embedding 嵌入模型以及向量数据库（Vector Store）。

LlamaIndex Python 库采用模块化命名空间规范：包含 `core` 的导入语句表示调用核心抽象包；不包含 `core` 的导入语句表示调用具体的生态集成插件。

```python
# 标准导入模式规范
from llama_index.core.xxx import ClassABC  # 核心抽象子模块 xxx
from llama_index.xxx.yyy import SubclassABC  # 子模块 xxx 的 yyy 具体集成实现

# 真实代码示例
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
```

### LlamaParse（文档智能体云平台）

**LlamaParse** 是一套专注于文档智能体（Document Agents）和智能 OCR 的独立平台。包含 **Parse**（文档解析）、**LlamaAgents**（智能体工作流）、**Extract**（结构化数据提取）以及 **Index**（数据摄取与 RAG 流水线）：

- **[注册 LlamaParse 账号](https://cloud.llamaindex.ai?utm_medium=li_github&utm_source=github&utm_campaign=2026--)** — 快速创建账号并获取 API Key。
- **Parse** — 智能体 OCR 与文档解析（支持 130+ 种格式）。[查看文档](https://developers.llamaindex.ai/python/cloud/llamaparse/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)
- **Extract** — 从复杂文档中自动化提取结构化字段。[查看文档](https://developers.llamaindex.ai/python/cloud/llamaextract/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)
- **Index** — 数据摄取、向量索引与端到端 RAG 管道。[查看文档](https://developers.llamaindex.ai/python/cloud/llamacloud/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)
- **Split** — 自动将超长文档智能拆分为业务子类别。[查看文档](https://developers.llamaindex.ai/python/cloud/split/getting_started/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)
- **Agents** — 基于 `Workflows` 与 Agent Builder 构建端到端文档智能体。[查看文档](https://developers.llamaindex.ai/python/llamaagents/overview/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)

### 常用链接 (Important Links)

- [官方开发者文档](https://developers.llamaindex.ai/python/framework/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)
- [X (原 Twitter)](https://x.com/llama_index)
- [LinkedIn 官方主页](https://www.linkedin.com/company/llamaindex/)
- [Reddit 社区](https://www.reddit.com/r/LlamaIndex/)
- [Discord 开发者频道](https://discord.gg/dGcwcsnxhU)

---

## 🚀 项目总览 (Overview)

### 背景痛点

- 大语言模型（LLM）在知识生成与逻辑推理方面展现了惊人能力，但它们仅基于公开数据预训练。
- **如何才能将大语言模型与企业/个人的私有业务数据进行最佳结合？**

我们需要一套完备的数据增强与连接工具箱。

### 解决方案

这就是 **LlamaIndex** 的核心使命。LlamaIndex 是专为大模型应用设计的数据框架，提供以下关键能力：

- **丰富的数据连接器（Data Connectors）**：直接摄取多种格式的私有数据源（API、PDF、Word、Markdown、SQL 数据库等）。
- **专业的数据结构化（Data Structuring）**：构建向量索引（Indices）、知识图谱（Knowledge Graphs）与层次化索引，便于 LLM 高效理解。
- **高级数据检索与查询接口（Advanced Retrieval / Query Engine）**：输入任意 Prompt 提示词，自动完成上下文检索并输出知识增强结果。
- **与外围框架轻松无缝集成**：可无缝配合 LangChain、Flask、Docker、ChatGPT 插件或任何自建系统使用。

LlamaIndex 同时服务于初学者与资深架构师：高阶 API 支持仅用 **5 行代码** 即可完成数据摄取与智能问答；低阶 API 则支持对数据连接器、索引结构、检索器、查询引擎与重排模块（Reranker）进行深度定制。

---

## 💻 代码实战示例 (Example Usage)

```bash
# 按需安装核心库与具体集成包
pip install llama-index-core
pip install llama-index-llms-openai
pip install llama-index-llms-ollama
pip install llama-index-embeddings-huggingface
```

### 1. 使用 OpenAI 快速构建向量索引

```python
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 加载指定目录下的全部文档并构建向量索引
documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(documents)
```

### 2. 使用本地开源大模型（如 Ollama + HuggingFace Embedding）

```python
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from transformers import AutoTokenizer

# 配置本地运行的 LLM 模型
Settings.llm = Ollama(
    model="llama-3.1:latest",
    request_timeout=360.0,
)

# 配置对应的分词器 Tokenizer
Settings.tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct"
)

# 配置开源本地向量嵌入模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(documents)
```

### 3. 执行智能检索与问答

```python
# 转换为查询引擎并执行提问
query_engine = index.as_query_engine()
response = query_engine.query("你的提问内容")
print(response)
```

### 4. 索引持久化与磁盘加载

默认情况下数据保存在内存中。如需持久化保存至本地磁盘（例如 `./storage` 目录）：

```python
# 持久化至磁盘
index.storage_context.persist(persist_dir="./storage")
```

从磁盘重新加载已构建的索引：

```python
from llama_index.core import StorageContext, load_index_from_storage

# 重建存储上下文并加载索引
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

---

## 💡 参与贡献 (Contributing)

想参与 LlamaIndex 贡献？无论是对核心库（Core）的改进，还是构建全新的集成插件，我们都由衷欢迎！详情请阅读 [贡献指南 (Contribution Guide)](CONTRIBUTING.md)。

## 📖 学术引用 (Citation)

如果您在科研论文或学术项目中使用了 LlamaIndex，请引用如下格式：

```bibtex
@software{Liu_LlamaIndex_2022,
author = {Liu, Jerry},
doi = {10.5281/zenodo.1234},
month = {11},
title = {{LlamaIndex}},
url = {https://github.com/jerryjliu/llama_index},
year = {2022}
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年8月31日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
