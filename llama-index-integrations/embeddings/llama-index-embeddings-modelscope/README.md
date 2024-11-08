# LlamaIndex Embedding Integration: ModelScope

## Installation

To install the required package, run:

```bash
!pip install llama-index-embeddings-modelscope
```

## Basic Usage

### Initialize the ModelScopeLLM

To use the ModelScopeEmbedding model, create an instance by specifying the model name and revision:

```python
from llama_index.embeddings.modelscope.base import ModelScopeEmbedding

model = ModelScopeEmbedding(
    model_name="iic/nlp_gte_sentence-embedding_chinese-base",
    model_revision="master",
)
```

### Generate Embedding

To generate a text embedding for a query, use the `get_query_embedding` method or `get_text_embedding` method:

```python
rsp = model.get_query_embedding("Hello, who are you?")
print(rsp)

rsp = model.get_text_embedding("Hello, who are you?")
print(rsp)
```

### Generate Batch Embedding

To generate a text embedding for a batch of text, use the `get_text_embedding_batch` method:

```python
rsp = model.get_text_embedding_batch(
    ["Hello, who are you?", "I am a student."]
)
print(rsp)
```
