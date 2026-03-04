# LlamaIndex Embeddings Integration: AutoEmbeddings

AutoEmbeddings is a very useful module available within [Chonkie](https://docs.chonkie.ai) that can initialize several different embeddings providers within one single interface:

- OpenAI
- Model2Vec
- Cohere
- Jina AI
- Sentence Transformers

You can install it with:

```bash
pip install llama-index-embeddings-autoembeddings
```

And then you can use it in your scripts as:

```python
from llama_index.embeddings.autoembeddings import ChonkieAutoEmbedding

embedder = ChonkieAutoEmbedding(model_name="all-MiniLM-L6-v2")
vector = embedder.get_text_embedding(
    "The quick brown fox jumps over the lazy dog."
)
print(vector)
```

If you want to use it with a non-local embeddings provider, you should declare the API key as an environment variable:

```python
from llama_index.embeddings.autoembeddings import ChonkieAutoEmbedding
import os

os.environ["OPENAI_API_KEY"] = "YOUR-API-KEY"
embedder = ChonkieAutoEmbedding(model_name="text-embedding-3-large")
vector = embedder.get_text_embedding(
    "The quick brown fox jumps over the lazy dog."
)
print(vector)
```
