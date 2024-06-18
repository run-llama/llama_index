# LlamaIndex Readers Integration: SteamshipFile

## Overview

The SteamshipFile Reader allows you to load documents from persistent Steamship Files. Steamship is a platform for storing and managing files with advanced tagging capabilities.

For more detailed information about the SteamshipFile Reader, visit [SteamShip](steamship.com).

### Installation

You can install the SteamshipFile Reader via pip:

```bash
pip install llama-index-readers-steamship
```

This reader requires steamship API key, which can be acquired from [SteamShip](steamship.com).

### Usage

```python
from llama_index.readers.steamship import SteamshipFileReader

# Initialize SteamshipFileReader
reader = SteamshipFileReader(api_key="<Steamship API Key>")

# Load data from persistent Steamship Files
documents = reader.load_data(
    workspace="<Steamship Workspace>",
    query="<Steamship Tag Query>",
    file_handles=["smooth-valley-9kbdr"],
    collapse_blocks=True,
    join_str="\n\n",
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
