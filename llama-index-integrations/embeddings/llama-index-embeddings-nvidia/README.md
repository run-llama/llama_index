# NVIDIA's Embeddings connector

With this connector, you'll be able to connect to and generate from compatible models available as hosted [NVIDIA NIMs](https://ai.nvidia.com), such as:

- NVIDIA's Retrieval QA Embedding Model [embed-qa-4](https://build.nvidia.com/nvidia/embed-qa-4)

_First_, get a free API key. Go to https://build.nvidia.com, select a model, click "Get API Key".
Store this key in your environment as `NVIDIA_API_KEY`.

## Installation

```bash
pip install llama-index-embeddings-nvidia
```

## Usage

```python
from llama_index.embeddings.nvidia import NVIDIAEmbedding

embedder = NVIDIAEmbedding()
embedder.get_query_embedding("What's the weather like in Komchatka?")
```
