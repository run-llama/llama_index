# 🗂️ LlamaIndex 🦙

LlamaIndex (GPT Index) is a project that provides a central interface to connect your LLM's with external data.

PyPI: 
- LlamaIndex: https://pypi.org/project/llama-index/.
- GPT Index (duplicate): https://pypi.org/project/gpt-index/.

Documentation: https://gpt-index.readthedocs.io/.

Twitter: https://twitter.com/gpt_index.

Discord: https://discord.gg/dGcwcsnxhU.

### Ecosystem

- LlamaHub (community library of data loaders): https://llamahub.ai
- LlamaLab (cutting-edge AGI projects using LlamaIndex): https://github.com/run-llama/llama-lab


## 🚀 Overview

**NOTE**: This README is not updated as frequently as the documentation. Please check out the documentation above for the latest updates!

### Context
- LLMs are a phenomenal piece of technology for knowledge generation and reasoning. They are pre-trained on large amounts of publicly available data.
- How do we best augment LLMs with our own private data?
- One paradigm that has emerged is *in-context* learning (the other is finetuning), where we insert context into the input prompt. That way,
we take advantage of the LLM's reasoning capabilities to generate a response.

To perform LLM's data augmentation in a performant, efficient, and cheap manner, we need to solve two components:
- Data Ingestion
- Data Indexing

### Proposed Solution

That's where the **LlamaIndex** comes in. LlamaIndex is a simple, flexible interface between your external data and LLMs. It provides the following tools in an easy-to-use fashion:

- Offers **data connectors** to your existing data sources and data formats (API's, PDF's, docs, SQL, etc.)
- Provides **indices** over your unstructured and structured data for use with LLM's. 
These indices help to abstract away common boilerplate and pain points for in-context learning:
   - Storing context in an easy-to-access format for prompt insertion.
   - Dealing with prompt limitations (e.g. 4096 tokens for Davinci) when context is too big.
   - Dealing with text splitting.
- Provides users an interface to **query** the index (feed in an input prompt) and obtain a knowledge-augmented output.
- Offers you a comprehensive toolset trading off cost and performance.


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
