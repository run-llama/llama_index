# 🗂️ ️GPT Index (LlamaIndex 🦙)

> ⚠️ **NOTE**: We are rebranding GPT Index as LlamaIndex! We will carry out this transition gradually.

> **2/27/2022**: We have a duplicate `llama-index` pip package. Simply replace all imports of `gpt_index` with `llama_index` if you choose to `pip install llama-index`.

GPT Index (LlamaIndex) is a project consisting of a set of data structures designed to make it easier to 
use large external knowledge bases with LLMs.

PyPi: 
- GPT Index: https://pypi.org/project/gpt-index/.
- LlamaIndex (duplicate): https://pypi.org/project/llama-index/.

Documentation: https://gpt-index.readthedocs.io/en/latest/.

Twitter: https://twitter.com/gpt_index.

Discord: https://discord.gg/dGcwcsnxhU.

LlamaHub (community library of data loaders): https://llamahub.ai

## 🚀 Overview

**NOTE**: This README is not updated as frequently as the documentation. Please check out the documentation above for the latest updates!

#### Context
- LLMs are a phenomenonal piece of technology for knowledge generation and reasoning.
- A big limitation of LLMs is context size (e.g. Davinci's limit is 4096 tokens. Large, but not infinite).
- The ability to feed "knowledge" to LLMs is restricted to this limited prompt size and model weights.

#### Proposed Solution

At its core, GPT Index contains a toolkit of **index data structures** designed to easily connect LLM's with your external data.
GPT Index helps to provide the following advantages:
- Remove concerns over prompt size limitations.
- Abstract common usage patterns to reduce boilerplate code in your LLM app.
- Provide data connectors to your common data sources (Google Docs, Slack, etc.).
- Provide cost transparency + tools that reduce cost while increasing performance.


Each data structure offers distinct use cases and a variety of customizable parameters. These indices can then be 
*queried* in a general purpose manner, in order to achieve any task that you would typically achieve with an LLM:
- Question-Answering
- Summarization
- Text Generation (Stories, TODO's, emails, etc.)
- and more!


## 💡 Contributing

Interesting in contributing? See our [Contribution Guide](CONTRIBUTING.md) for more details.

## 📄 Documentation

Full documentation can be found here: https://gpt-index.readthedocs.io/en/latest/. 

Please check it out for the most up-to-date tutorials, how-to guides, references, and other resources! 


## 💻 Example Usage

```
pip install gpt-index
```

Examples are in the `examples` folder. Indices are in the `indices` folder (see list of indices below).

To build a simple vector store index:
```python
import os
os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_API_KEY'

from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex(documents)
```

To save to and load from disk:
```python
# save to disk
index.save_to_disk('index.json')
# load from disk
index = GPTSimpleVectorIndex.load_from_disk('index.json')
```

To query:
```python
index.query("<question_text>?")
```

## 🔧 Dependencies

The main third-party package requirements are `tiktoken`, `openai`, and `langchain`.

All requirements should be contained within the `setup.py` file. To run the package locally without building the wheel, simply run `pip install -r requirements.txt`. 


## 📖 Citation

Reference to cite if you use GPT Index in a paper:

```
@software{Liu_GPT_Index_2022,
author = {Liu, Jerry},
doi = {10.5281/zenodo.1234},
month = {11},
title = {{GPT Index}},
url = {https://github.com/jerryjliu/gpt_index},year = {2022}
}
```
