# llama-index-postprocessor-gpubridge-rerank

GPU-Bridge reranker integration for LlamaIndex.

## Install

```bash
pip install llama-index-postprocessor-gpubridge-rerank
```

## Usage

```python
from llama_index.postprocessor.gpubridge_rerank import GPUBridgeRerank

reranker = GPUBridgeRerank(api_key="gpub_...", top_n=3)
nodes = reranker.postprocess_nodes(nodes, query_bundle=query)
```

Get an API key at https://gpubridge.xyz
