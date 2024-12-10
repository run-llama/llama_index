# LlamaIndex Embeddings Integration: Octoai

Using the [OctoAI](https://octo.ai) Embeddings Integration is a simple as:

```python
from llama_index.embeddings.octoai import OctoAIEmbedding
from os import environ

OCTOAI_API_KEY = environ["OCTOAI_API_KEY"]
embed_model = OctoAIEmbedding(api_key=OCTOAI_API_KEY)
embeddings = embed_model.get_text_embedding("How do I sail to the moon?")
assert len(embeddings) == 1024
```

One can also request a batch of embeddings via:

```python
texts = [
    "How do I sail to the moon?",
    "What is the best way to cook a steak?",
    "How do I apply for a job?",
]

embeddings = embed_model.get_text_embedding_batch(texts)
assert len(embeddings) == 3
```

## API Access

[Here](https://octo.ai/docs/getting-started/how-to-create-an-octoai-access-token) are some instructions on how to get your OctoAI API key.

## Contributing

Follow the good practices of all poetry based projects.

When in VScode, one may want to manually select the Python interpreter, specially to run the example iPython notebook. For this use `ctrl+shift+p`, then type or select: `Python: Select Interpreter`
