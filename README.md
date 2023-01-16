# 🗂️ ️GPT Index

GPT Index is a project consisting of a set of *data structures* that are created using LLMs and can be traversed using LLMs in order to answer queries.

PyPi: https://pypi.org/project/gpt-index/.

Documentation: https://gpt-index.readthedocs.io/en/latest/.

Discord: https://discord.gg/dGcwcsnxhU.

## 🚀 Overview

**NOTE**: This README is not updated as frequently as the documentation. Please check out the documentation above for the latest updates!

#### Context
- LLMs are a phenomenonal piece of technology for knowledge generation and reasoning.
- A big limitation of LLMs is context size (e.g. Davinci's limit is 4096 tokens. Large, but not infinite).
- The ability to feed "knowledge" to LLMs is restricted to this limited prompt size and model weights.

#### Proposed Solution
That's where the **GPT Index** comes in. GPT Index is a simple, flexible interface between your external data and LLMs. It resolves the following pain points:

- Provides simple data structures to resolve prompt size limitations.
- Offers data connectors to your external data sources.
- Offers you a comprehensive toolset trading off cost and performance.

At the core of GPT Index is a **data structure**. Instead of relying on world knowledge encoded in the model weights, a GPT Index data structure does the following:

- Uses a pre-trained LLM primarily for *reasoning*/*summarization* instead of prior knowledge.
- Takes as input a large corpus of text data and build a structured index over it (using an LLM or heuristics).
- Allow users to *query* the index by passing in an LLM prompt, and obtaining a response.

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
index.query("<question_text>?", child_branch_factor=1)
```

## 🔧 Dependencies

The main third-party package requirements are `tiktoken`, `openai`, and `langchain`.

All requirements should be contained within the `setup.py` file. To run the package locally without building the wheel, simply run `pip install -r requirements.txt`. 


