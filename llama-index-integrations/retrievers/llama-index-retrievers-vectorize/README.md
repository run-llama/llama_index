# LlamaIndex Retrievers Integration: Vectorize

> [Vectorize](https://vectorize.io/) RAG-as-a-Service handles the messy, hard parts of AI development,
> so you can focus on building your applications.

## Installation

```bash
pip install llama-index-retrievers-vectorize
```

### Usage

```python
from llama_index.retrievers.vectorize import VectorizeRetriever

retriever = VectorizeRetriever(
    api_token="...",
    organization="...",
    pipeline_id="...",
)
retriever.retrieve("query")
```
