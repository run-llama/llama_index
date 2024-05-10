# LlamaIndex Readers Integration: Jaguar

## Overview

Jaguar Reader retrieves documents from an existing persisted Jaguar store. These documents can then be used in a downstream LlamaIndex data structure.

### Installation

You can install Jaguar Reader via pip:

```bash
pip install llama-index-readers-jaguar
```

To use Jaguar Reader, you must have an API key. Here are the [installation instructions](http://www.jaguardb.com/docsetup.html)

## Usage

```python
from llama_index.readers.jaguar import JaguarReader

# Initialize JaguarReader
reader = JaguarReader(
    pod="<Pod Name>",
    store="<Store Name>",
    vector_index="<Vector Index Name>",
    vector_type="<Vector Type>",
    vector_dimension="<Vector Dimension>",
    url="<Endpoint URL>",
)

# Login to Jaguar server
reader.login(jaguar_api_key="<Jaguar API Key>")

# Load data from Jaguar
documents = reader.load_data(
    embedding="<Embedding Vector>",
    k=10,
    metadata_fields=["<Metadata Field 1>", "<Metadata Field 2>"],
    where="<Query Condition>",
)

# Logout from Jaguar server
reader.logout()
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).
