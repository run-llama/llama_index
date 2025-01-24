# 🗂️ LlamaIndex 🦙

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-index)](https://pypi.org/project/llama-index/)
[![GitHub contributors](https://img.shields.io/github/contributors/jerryjliu/llama_index)](https://github.com/jerryjliu/llama_index/graphs/contributors)
[![Discord](https://img.shields.io/discord/1059199217496772688)](https://discord.gg/dGcwcsnxhU)
[![Ask AI](https://img.shields.io/badge/Phorm-Ask_AI-%23F2777A.svg?&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNSIgaGVpZ2h0PSI0IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogIDxwYXRoIGQ9Ik00LjQzIDEuODgyYTEuNDQgMS40NCAwIDAgMS0uMDk4LjQyNmMtLjA1LjEyMy0uMTE1LjIzLS4xOTIuMzIyLS4wNzUuMDktLjE2LjE2NS0uMjU1LjIyNmExLjM1MyAxLjM1MyAwIDAgMS0uNTk1LjIxMmMtLjA5OS4wMTItLjE5Mi4wMTQtLjI3OS4wMDZsLTEuNTkzLS4xNHYtLjQwNmgxLjY1OGMuMDkuMDAxLjE3LS4xNjkuMjQ2LS4xOTFhLjYwMy42MDMgMCAwIDAgLjItLjEwNi41MjkuNTI5IDAgMCAwIC4xMzgtLjE3LjY1NC42NTQgMCAwIDAgLjA2NS0uMjRsLjAyOC0uMzJhLjkzLjkzIDAgMCAwLS4wMzYtLjI0OS41NjcuNTY3IDAgMCAwLS4xMDMtLjIuNTAyLjUwMiAwIDAgMC0uMTY4LS4xMzguNjA4LjYwOCAwIDAgMC0uMjQtLjA2N0wyLjQzNy43MjkgMS42MjUuNjcxYS4zMjIuMzIyIDAgMCAwLS4yMzIuMDU4LjM3NS4zNzUgMCAwIDAtLjExNi4yMzJsLS4xMTYgMS40NS0uMDU4LjY5Ny0uMDU4Ljc1NEwuNzA1IDRsLS4zNTctLjA3OUwuNjAyLjkwNkMuNjE3LjcyNi42NjMuNTc0LjczOS40NTRhLjk1OC45NTggMCAwIDEgLjI3NC0uMjg1Ljk3MS45NzEgMCAwIDEgLjMzNy0uMTRjLjExOS0uMDI2LjIyNy0uMDM0LjMyNS0uMDI2TDMuMjMyLjE2Yy4xNTkuMDE0LjMzNi4wMy40NTkuMDgyYTEuMTczIDEuMTczIDAgMCAxIC41NDUuNDQ3Yy4wNi4wOTQuMTA5LjE5Mi4xNDQuMjkzYTEuMzkyIDEuMzkyIDAgMCAxIC4wNzguNThsLS4wMjkuMzJaIiBmaWxsPSIjRjI3NzdBIi8+CiAgPHBhdGggZD0iTTQuMDgyIDIuMDA3YTEuNDU1IDEuNDU1IDAgMCAxLS4wOTguNDI3Yy0uMDUuMTI0LS4xMTQuMjMyLS4xOTIuMzI0YTEuMTMgMS4xMyAwIDAgMS0uMjU0LjIyNyAxLjM1MyAxLjM1MyAwIDAgMS0uNTk1LjIxNGMtLjEuMDEyLS4xOTMuMDE0LS4yOC4wMDZsLTEuNTYtLjEwOC4wMzQtLjQwNi4wMy0uMzQ4IDEuNTU5LjE1NGMuMDkgMCAuMTczLS4wMS4yNDgtLjAzM2EuNjAzLjYwMyAwIDAgMCAuMi0uMTA2LjUzMi41MzIgMCAwIDAgLjEzOS0uMTcyLjY2LjY2IDAgMCAwIC4wNjQtLjI0MWwuMDI5LS4zMjFhLjk0Ljk0IDAgMCAwLS4wMzYtLjI1LjU3LjU3IDAgMCAwLS4xMDMtLjIwMi41MDIuNTAyIDAgMCAwLS4xNjgtLjEzOC42MDUuNjA1IDAgMCAwLS4yNC0uMDY3TDEuMjczLjgyN2MtLjA5NC0uMDA4LS4xNjguMDEtLjIyMS4wNTUtLjA1My4wNDUtLjA4NC4xMTQtLjA5Mi4yMDZMLjcwNSA0IDAgMy45MzhsLjI1NS0yLjkxMUExLjAxIDEuMDEgMCAwIDEgLjM5My41NzIuOTYyLjk2MiAwIDAgMSAuNjY2LjI4NmEuOTcuOTcgMCAwIDEgLjMzOC0uMTRDMS4xMjIuMTIgMS4yMy4xMSAxLjMyOC4xMTlsMS41OTMuMTRjLjE2LjAxNC4zLjA0Ny40MjMuMWExLjE3IDEuMTcgMCAwIDEgLjU0NS40NDhjLjA2MS4wOTUuMTA5LjE5My4xNDQuMjk1YTEuNDA2IDEuNDA2IDAgMCAxIC4wNzcuNTgzbC0uMDI4LjMyMloiIGZpbGw9IndoaXRlIi8+CiAgPHBhdGggZD0iTTQuMDgyIDIuMDA3YTEuNDU1IDEuNDU1IDAgMCAxLS4wOTguNDI3Yy0uMDUuMTI0LS4xMTQuMjMyLS4xOTIuMzI0YTEuMTMgMS4xMyAwIDAgMS0uMjU0LjIyNyAxLjM1MyAxLjM1MyAwIDAgMS0uNTk1LjIxNGMtLjEuMDEyLS4xOTMuMDE0LS4yOC4wMDZsLTEuNTYtLjEwOC4wMzQtLjQwNi4wMy0uMzQ4IDEuNTU5LjE1NGMuMDkgMCAuMTczLS4wMS4yNDgtLjAzM2EuNjAzLjYwMyAwIDAgMCAuMi0uMTA2LjUzMi41MzIgMCAwIDAgLjEzOS0uMTcyLjY2LjY2IDAgMCAwIC4wNjQtLjI0MWwuMDI5LS4zMjFhLjk0Ljk0IDAgMCAwLS4wMzYtLjI1LjU3LjU3IDAgMCAwLS4xMDMtLjIwMi41MDIuNTAyIDAgMCAwLS4xNjgtLjEzOC42MDUuNjA1IDAgMCAwLS4yNC0uMDY3TDEuMjczLjgyN2MtLjA5NC0uMDA4LS4xNjguMDEtLjIyMS4wNTUtLjA1My4wNDUtLjA4NC4xMTQtLjA5Mi4yMDZMLjcwNSA0IDAgMy45MzhsLjI1NS0yLjkxMUExLjAxIDEuMDEgMCAwIDEgLjM5My41NzIuOTYyLjk2MiAwIDAgMSAuNjY2LjI4NmEuOTcuOTcgMCAwIDEgLjMzOC0uMTRDMS4xMjIuMTIgMS4yMy4xMSAxLjMyOC4xMTlsMS41OTMuMTRjLjE2LjAxNC4zLjA0Ny40MjMuMWExLjE3IDEuMTcgMCAwIDEgLjU0NS40NDhjLjA2MS4wOTUuMTA5LjE5My4xNDQuMjk1YTEuNDA2IDEuNDA2IDAgMCAxIC4wNzcuNTgzbC0uMDI4LjMyMloiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo=)](https://www.phorm.ai/query?projectId=c5863b56-6703-4a5d-87b6-7e6031bf16b6)

LlamaIndex (GPT Index) is a data framework for your LLM application. Building with LlamaIndex typically involves working with LlamaIndex core and a chosen set of integrations (or plugins). There are two ways to start building with LlamaIndex in
Python:

1. **Starter**: [`llama-index`](https://pypi.org/project/llama-index/). A starter Python package that includes core LlamaIndex as well as a selection of integrations.

2. **Customized**: [`llama-index-core`](https://pypi.org/project/llama-index-core/). Install core LlamaIndex and add your chosen LlamaIndex integration packages on [LlamaHub](https://llamahub.ai/)
   that are required for your application. There are over 300 LlamaIndex integration
   packages that work seamlessly with core, allowing you to build with your preferred
   LLM, embedding, and vector store providers.

The LlamaIndex Python library is namespaced such that import statements which
include `core` imply that the core package is being used. In contrast, those
statements without `core` imply that an integration package is being used.

```python
# typical pattern
from llama_index.core.xxx import ClassABC  # core submodule xxx
from llama_index.xxx.yyy import (
    SubclassABC,
)  # integration yyy for submodule xxx

# concrete example
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
```

### Important Links

LlamaIndex.TS [(Typescript/Javascript)](https://github.com/run-llama/LlamaIndexTS)

[Documentation](https://docs.llamaindex.ai/en/stable/)

[X (formerly Twitter)](https://x.com/llama_index)

[Discord](https://discord.gg/dGcwcsnxhU)

### Ecosystem

- LlamaHub [(community library of data loaders)](https://llamahub.ai)
- LlamaLab [(cutting-edge AGI projects using LlamaIndex)](https://github.com/run-llama/llama-lab)

## 🚀 Overview

**NOTE**: This README is not updated as frequently as the documentation. Please check out the documentation above for the latest updates!

### Context

- LLMs are a phenomenal piece of technology for knowledge generation and reasoning. They are pre-trained on large amounts of publicly available data.
- How do we best augment LLMs with our own private data?

We need a comprehensive toolkit to help perform this data augmentation for LLMs.

### Proposed Solution

That's where **LlamaIndex** comes in. LlamaIndex is a "data framework" to help you build LLM apps. It provides the following tools:

- Offers **data connectors** to ingest your existing data sources and data formats (APIs, PDFs, docs, SQL, etc.).
- Provides ways to **structure your data** (indices, graphs) so that this data can be easily used with LLMs.
- Provides an **advanced retrieval/query interface over your data**: Feed in any LLM input prompt, get back retrieved context and knowledge-augmented output.
- Allows easy integrations with your outer application framework (e.g. with LangChain, Flask, Docker, ChatGPT, or anything else).

LlamaIndex provides tools for both beginner users and advanced users. Our high-level API allows beginner users to use LlamaIndex to ingest and query their data in
5 lines of code. Our lower-level APIs allow advanced users to customize and extend any module (data connectors, indices, retrievers, query engines, reranking modules),
to fit their needs.

## 💡 Contributing

Interested in contributing? Contributions to LlamaIndex core as well as contributing
integrations that build on the core are both accepted and highly encouraged! See our [Contribution Guide](CONTRIBUTING.md) for more details.

New integrations should meaningfully integrate with existing LlamaIndex framework components. At the discretion of LlamaIndex maintainers, some integrations may be declined.

## 📄 Documentation

Full documentation can be found [here](https://docs.llamaindex.ai/en/latest/)

Please check it out for the most up-to-date tutorials, how-to guides, references, and other resources!

## 💻 Example Usage

```sh
# custom selection of integrations to work with core
pip install llama-index-core
pip install llama-index-llms-openai
pip install llama-index-llms-replicate
pip install llama-index-embeddings-huggingface
```

Examples are in the `docs/examples` folder. Indices are in the `indices` folder (see list of indices below).

To build a simple vector store index using OpenAI:

```python
import os

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(documents)
```

To build a simple vector store index using non-OpenAI LLMs, e.g. Llama 2 hosted on [Replicate](https://replicate.com/), where you can easily create a free trial API token:

```python
import os

os.environ["REPLICATE_API_TOKEN"] = "YOUR_REPLICATE_API_TOKEN"

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from transformers import AutoTokenizer

# set the LLM
llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
Settings.llm = Replicate(
    model=llama2_7b_chat,
    temperature=0.01,
    additional_kwargs={"top_p": 1, "max_new_tokens": 300},
)

# set tokenizer to match LLM
Settings.tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf"
)

# set the embed model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(
    documents,
)
```

To query:

```python
query_engine = index.as_query_engine()
query_engine.query("YOUR_QUESTION")
```

By default, data is stored in-memory.
To persist to disk (under `./storage`):

```python
index.storage_context.persist()
```

To reload from disk:

```python
from llama_index.core import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")
# load index
index = load_index_from_storage(storage_context)
```

## 🔧 Dependencies

We use poetry as the package manager for all Python packages. As a result, the
dependencies of each Python package can be found by referencing the `pyproject.toml`
file in each of the package's folders.

```bash
cd <desired-package-folder>
pip install poetry
poetry install --with dev
```

## 📖 Citation

Reference to cite if you use LlamaIndex in a paper:

```
@software{Liu_LlamaIndex_2022,
author = {Liu, Jerry},
doi = {10.5281/zenodo.1234},
month = {11},
title = {{LlamaIndex}},
url = {https://github.com/jerryjliu/llama_index},
year = {2022}
}
```
