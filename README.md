# 🗂️ LlamaIndex 🦙

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-index)](https://pypi.org/project/llama-index/)
[![Build](https://github.com/run-llama/llama_index/actions/workflows/build_package.yml/badge.svg)](https://github.com/run-llama/llama_index/actions/workflows/build_package.yml)
[![GitHub contributors](https://img.shields.io/github/contributors/jerryjliu/llama_index)](https://github.com/jerryjliu/llama_index/graphs/contributors)
[![Discord](https://img.shields.io/discord/1059199217496772688)](https://discord.gg/dGcwcsnxhU)
[![Twitter](https://img.shields.io/twitter/follow/llama_index)](https://x.com/llama_index)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/LlamaIndex?style=plastic&logo=reddit&label=r%2FLlamaIndex&labelColor=white)](https://www.reddit.com/r/LlamaIndex/)

LlamaIndex OSS (by [LlamaIndex](https://llamaindex.ai?utm_medium=li_github&utm_source=github&utm_campaign=2026--)) is an open-source framework to build agentic applications. **[Parse](https://cloud.llamaindex.ai?utm_medium=li_github&utm_source=github&utm_campaign=2026--)** is our enterprise platform for agentic OCR, parsing, extraction, indexing and more. You can use LlamaParse with this framework or on its own; see [LlamaParse](#llamacloud-document-agent-platform) below for signup and product links.

> ### 📚 **Documentation:**
>
> - [LlamaParse](https://developers.llamaindex.ai/python/cloud/llamaparse/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)
> - [LlamaIndex OSS](https://developers.llamaindex.ai/python/framework/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)
> - [LlamaAgents](https://developers.llamaindex.ai/python/llamaagents/overview/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)

Building with LlamaIndex typically involves working with LlamaIndex core and a chosen set of integrations (or plugins). There are two ways to start building with LlamaIndex in
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

### LlamaParse (document agent platform)

**LlamaParse** is its own platform—focused on document agents and agentic OCR. It includes **Parse** (parsing), **LlamaAgents** (deployed document agents), **Extract** (structured extraction), and **Index** (ingest and RAG). You can use it with the LlamaIndex framework or standalone.

- **[Sign up for LlamaParse](https://cloud.llamaindex.ai?utm_medium=li_github&utm_source=github&utm_campaign=2026--)** — Create an account and get your API key.
- **Parse** — Agentic OCR and document parsing (130+ formats). [Docs](https://developers.llamaindex.ai/python/cloud/llamaparse/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)
- **Extract** — Structured data extraction from documents. [Docs](https://developers.llamaindex.ai/python/cloud/llamaextract/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)
- **Index** — Ingest, index, and RAG pipelines. [Docs](https://developers.llamaindex.ai/python/cloud/llamacloud/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)
- **Split** — Split large documents into subcategories. [Docs](https://developers.llamaindex.ai/python/cloud/split/getting_started/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)
- **Agents** — Build end-to-end document agents with `Workflows` and Agent Builder. [Docs](https://developers.llamaindex.ai/python/llamaagents/overview/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)

### Important Links

[Documentation](https://developers.llamaindex.ai/python/framework/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)

[X (formerly Twitter)](https://x.com/llama_index)

[LinkedIn](https://www.linkedin.com/company/llamaindex/)

[Reddit](https://www.reddit.com/r/LlamaIndex/)

[Discord](https://discord.gg/dGcwcsnxhU)

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

Full documentation can be found [here](https://developers.llamaindex.ai/python/framework/?utm_medium=li_github&utm_source=github&utm_campaign=2026--)

Please check it out for the most up-to-date tutorials, how-to guides, references, and other resources!

## 💻 Example Usage

```sh
# custom selection of integrations to work with core
pip install llama-index-core
pip install llama-index-llms-openai
pip install llama-index-llms-ollama
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

To build a simple vector store index using non-OpenAI LLMs, e.g. LLMs hosted through Ollama:

```python
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from transformers import AutoTokenizer

# set the LLM
Settings.llm = Ollama(
    model="llama-3.1:latest",
    request_timeout=360.0,
)

# set tokenizer to match LLM
Settings.tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct"
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

## A note on Verification of Build Assets

By default, `llama-index-core` includes a `_static` folder that contains the nltk and tiktoken cache that is included with the package installation. This ensures that you can easily run `llama-index` in environments with restrictive disk access permissions at runtime.

To verify that these files are safe and valid, we use the github `attest-build-provenance` action. This action will verify that the files in the `_static` folder are the same as the files in the `llama-index-core/llama_index/core/_static` folder.

To verify this, you can run the following script (pointing to your installed package):

```bash
#!/bin/bash
STATIC_DIR="venv/lib/python3.13/site-packages/llama_index/core/_static"
REPO="run-llama/llama_index"

find "$STATIC_DIR" -type f | while read -r file; do
    echo "Verifying: $file"
    gh attestation verify "$file" -R "$REPO" || echo "Failed to verify: $file"
done
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
