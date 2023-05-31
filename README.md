# 🗂️ LlamaIndex 🦙

LlamaIndex (GPT Index) is a data framework for your LLM application.

PyPI: 
- LlamaIndex: https://pypi.org/project/llama-index/.
- GPT Index (duplicate): https://pypi.org/project/gpt-index/.

Documentation: https://gpt-index.readthedocs.io/.

Twitter: https://twitter.com/llama_index.

Discord: https://discord.gg/dGcwcsnxhU.

### Ecosystem

- LlamaHub (community library of data loaders): https://llamahub.ai
- LlamaLab (cutting-edge AGI projects using LlamaIndex): https://github.com/run-llama/llama-lab


## 🚀 Overview

**NOTE**: This README is not updated as frequently as the documentation. Please check out the documentation above for the latest updates!

### Context
- LLMs are a phenomenonal piece of technology for knowledge generation and reasoning. They are pre-trained on large amounts of publicly available data.
- How do we best augment LLMs with our own private data?

We need a comprehensive toolkit to help perform this data augmentation for LLMs.

### Proposed Solution

That's where **LlamaIndex** comes in. LlamaIndex is a "data framework" to help you build LLM apps. It provides the following tools:

- Offers **data connectors** to ingest your existing data sources and data formats (APIs, PDFs, docs, SQL, etc.)
- Provides ways to **structure your data** (indices, graphs) so that this data can be easily used with LLMs.
- Provides an **advanced retrieval/query interface over your data**: Feed in any LLM input prompt, get back retrieved context and knowledge-augmented output.
- Allows easy integrations with your outer application framework (e.g. with LangChain, Flask, Docker, ChatGPT, anything else).

LlamaIndex provides tools for both beginner users and advanced users. Our high-level API allows beginner users to use LlamaIndex to ingest and query their data in
5 lines of code. Our lower-level APIs allow advanced users to customize and extend any module (data connectors, indices, retrievers, query engines, reranking modules),
to fit their needs.


## 💡 Contributing

Interested in contributing? See our [Contribution Guide](CONTRIBUTING.md) for more details.

## 📄 Documentation

Full documentation can be found here: https://gpt-index.readthedocs.io/en/latest/. 

Please check it out for the most up-to-date tutorials, how-to guides, references, and other resources! 


## 💻 Example Usage

```
pip install llama-index
```

Examples are in the `examples` folder. Indices are in the `indices` folder (see list of indices below).

To build a simple vector store index:
```python
import os
os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_API_KEY'

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)
```


To query:
```python
query_engine = index.as_query_engine()
query_engine.query("<question_text>?")
```


By default, data is stored in-memory.
To persist to disk (under `./storage`):

```python
index.storage_context.persist()
```

To reload from disk:
```python
from llama_index import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir='./storage')
# load index
index = load_index_from_storage(storage_context)
```


## 🔧 Dependencies

The main third-party package requirements are `tiktoken`, `openai`, and `langchain`.

All requirements should be contained within the `setup.py` file. To run the package locally without building the wheel, simply run `pip install -r requirements.txt`. 


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
