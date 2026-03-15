# llama-index-embeddings-gpubridge

GPU-Bridge embeddings integration for LlamaIndex.

## Install

```bash
pip install llama-index-embeddings-gpubridge
```

## Usage

```python
from llama_index.embeddings.gpubridge import GPUBridgeEmbedding

embed_model = GPUBridgeEmbedding(api_key="gpub_...")
vectors = embed_model.get_text_embedding_batch(["text 1", "text 2"])
```

Get an API key at https://gpubridge.xyz
