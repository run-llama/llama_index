# LlamaIndex Readers Integration: StringIterable

## Overview

The StringIterable Reader converts an iterable of strings into a list of documents. It's a simple utility to quickly create documents from a list of text strings.

### Installation

You can install String Iterable Reader via pip:

```bash
pip install llama-index-readers-string-iterable
```

## Usage

```python
from llama_index.readers.string_iterable import StringIterableReader

# Initialize StringIterableReader
reader = StringIterableReader()

# Load data from an iterable of strings
documents = reader.load_data(
    texts=["I went to the store", "I bought an apple"]
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
