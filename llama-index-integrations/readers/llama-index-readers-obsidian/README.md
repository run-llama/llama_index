# LlamaIndex Readers Integration: Obsidian

## Overview

Obsidian Reader is a utility class for loading data from an Obsidian Vault. It parses all markdown files in the vault and extracts text from under Obsidian headers, returning a list of documents, each containing text from a separate header.

### Usage

```python
from llama_index.readers.obsidian import ObsidianReader

# Initialize ObsidianReader with the path to the Obsidian vault
reader = ObsidianReader(input_dir="<Path to Obsidian Vault>")

# Load data from the Obsidian vault
documents = reader.load_data()
```

Implementation for Obsidian reader can be found [here](https://docs.llamaindex.ai/en/stable/examples/data_connectors/ObsidianReaderDemo/)

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
