# LlamaIndex Readers Integration: Quip

## Overview

The Quip Reader enables loading data from Quip documents. It constructs queries to retrieve thread content based on thread IDs.

### Installation

You can install the Quip Reader via pip:

```bash
pip install llama-index-readers-quip
```

### Usage

```python
from llama_index.readers.quip import QuipReader

# Initialize QuipReader
reader = QuipReader(access_token="<Access Token>")

# Load data from Quip
documents = reader.load_data(thread_ids=["<Thread ID 1>", "<Thread ID 2>"])
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
