# Redis Semantic Cache

Traditional caching relies on exact string matches (key-value pairs). If a user asks "How do I reset my password?" and another asks "Password reset steps?", a traditional cache treats them as entirely different queries.

Semantic Caching uses Vector Similarity Search to understand the meaning behind a query. Instead of looking for identical text, it looks for queries that are "close enough" in meaning to return a previously generated response.

Semantic caching delivers real advantages for LLM-powered apps as RAGs:

- **Faster responses:** Cached responses return in milliseconds instead of the seconds it takes for an LLM to generate a fresh answer. For high-traffic apps, this difference is everything. Users get instant replies, and your system handles more concurrent requests without breaking a sweat.
- **Lower costs**: LLM API calls add up fast. Every time you hit the cache instead of calling the model, you save money. Teams using semantic caching typically cut their LLM costs by 50% or more, depending on how repetitive their query patterns are. The more similar questions your users ask, the bigger the savings.

## Getting started

### Run Redis Locally

```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

### Initialize your embegging model

```python
import os
from llama_index.embeddings.openai import OpenAIEmbedding

os.environ["OPENAI_API_KEY"] = "sk-..."

embed_model = OpenAIEmbedding(model="text-embedding-3-large")
embedding_dims = len(embed_model.get_text_embedding("test text"))
```

### Initialize Semantic Cache

```python
cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embed_model=embed_model,
    embedding_dims=embedding_dims,
    name="semanti_cache",
    prefix="semantic_cache_prefix",
    ttl=3600,
)
```

### Add an entry to the cache

```python
cache.store_cache_entry(
    query="Where did Galileo Galilei teach?",
    response="Galileo Galilei taught at the University of Padua.",
    metadata={"source": "my_story_book.pdf"},
)
```

### Get match result from cache

```python
result = cache.check("In which city did Galileo Galilei teach?")
```

you will get a result similar to:

```python
CacheResults(
    matches=[
        CacheResult(
            id="4758e5c55ed38ca280a0c8bc545983d49e5b3ca42b98c052230958e087339ca9",
            key="semantic_cache:4758e5c55ed38ca280a0c8bc545983d49e5b3ca42b98c052230958e087339ca9",
            query="Where did Galileo Galilei teach?",
            response="Galileo Galilei taught at the University of Padua.",
            metadata={"source": "my_story_book.pdf"},
            vector_distance=0.0230112075806,
            cosine_similarity=0.9884943962097,
            inserted_at=1774953382.47,
            updated_at=1774953382.47,
        )
    ]
)
```

### Clean the cache

```python
cache.clean_cache()
```

### Run the tests

```bash
uv run pytest
```
