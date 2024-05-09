# LlamaIndex Readers Integration: Json

## Overview

JSON Reader reads JSON documents with options to help extract relationships between nodes. It provides functionalities to control the depth of JSON traversal, collapse long JSON fragments, and clean JSON structures.

### Installation

You can install JSON Reader via pip:

```bash
pip install llama-index-readers-json
```

## Usage

```python
from llama_index.readers.json import JSONReader

# Initialize JSONReader
reader = JSONReader(
    # The number of levels to go back in the JSON tree. Set to 0 to traverse all levels. Default is None.
    levels_back="<Levels Back>",
    # The maximum number of characters a JSON fragment would be collapsed in the output. Default is None.
    collapse_length="<Collapse Length>",
    # If True, ensures that the output is ASCII-encoded. Default is False.
    ensure_ascii="<Ensure ASCII>",
    # If True, indicates that the file is in JSONL (JSON Lines) format. Default is False.
    is_jsonl="<Is JSONL>",
    # If True, removes lines containing only formatting from the output. Default is True.
    clean_json="<Clean JSON>",
)

# Load data from JSON file
documents = reader.load_data(input_file="<Input File>", extra_info={})
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
