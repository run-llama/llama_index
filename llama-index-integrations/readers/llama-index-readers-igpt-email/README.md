# LlamaIndex Readers Integration: iGPT Email Intelligence

```bash
pip install llama-index-readers-igpt-email
```

The iGPT Email Intelligence Reader loads structured, reasoning-ready email
context from the iGPT API as LlamaIndex Documents for indexing and retrieval.

Unlike raw email connectors that return unprocessed message data, iGPT handles
thread reconstruction, participant role detection, and intent extraction before
returning results — so each Document contains clean, structured content ready
for a RAG pipeline.

To begin, you need to obtain an API key at [docs.igpt.ai](https://docs.igpt.ai).

## Usage

Here's an example usage of the IGPTEmailReader.

```python
from llama_index.readers.igpt_email import IGPTEmailReader
from llama_index.core import VectorStoreIndex

reader = IGPTEmailReader(api_key="your-key", user="user-id")
documents = reader.load_data(query="project Alpha", date_from="2025-01-01")
index = VectorStoreIndex.from_documents(documents)
```

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
