# LlamaIndex Readers Integration: Psychic

## Overview

The Psychic Reader facilitates the extraction of data from the Psychic platform, enabling seamless access to synced data from various SaaS applications through a single universal API.
Psychic is a platform designed to synchronize data from multiple SaaS applications via a universal API. This reader module connects to a Psychic instance and retrieves data, requiring authentication via a secret key.

For more detailed information about Psychic, visit [docs.psychic.dev](https://docs.psychic.dev).

### Installation

You can install Psychic Reader via pip:

```bash
pip install llama-index-readers-psychic
```

### Usage

```python
from llama_index.readers.psychic import PsychicReader

# Initialize PsychicReader
reader = PsychicReader(psychic_key="<Psychic Secret Key>")

# Load data from Psychic
documents = reader.load_data(
    connector_id="<Connector ID>", account_id="<Account ID>"
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index) and/or subsequently
used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
