# LLamaIndex node postprocessing reranker using pinecone hosted models

- use rerank models with the pinecone managed vector service to rerank the search results
- available rerank models from [pinecone](https://app.pinecone.io/organizations/-Nn577_974iRsvC6nVxg/projects/a4fe57a4-b1cc-4a99-bf1d-c35a595cae4a/models)

```python
import os
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.postprocessor.pinecone_native_rerank import (
    PineconeNativeRerank,
)
from llama_index.core.response.pprint_utils import pprint_response

os.environ["PINECONE_API_KEY"] = "your_pinecone_api_key"
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

txts = [
    "Apple is a popular fruit known for its sweetness and crisp texture.",
    "Apple is known for its innovative products like the iPhone.",
    "Many people enjoy eating apples as a healthy snack.",
    "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
    "An apple a day keeps the doctor away, as the saying goes.",
    "apple has a lot of vitamins",
]

nodes = [TextNode(id_=f"vec{i}", text=txt) for i, txt in enumerate(txts)]

pinecone_reranker = PineconeNativeRerank(top_n=4, model="pinecone-rerank-v0")

index = VectorStoreIndex(nodes)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[pinecone_reranker],
)

response = query_engine.query(
    "The tech company Apple is known for its innovative products like the iPhone."
)

pprint_response(response, show_source=True)
```

output

```txt
Final Response: Apple is recognized for its innovative products like
the iPhone.
______________________________________________________________________
Source Node 1/4
Node ID: vec1
Similarity: 0.9655668
Text: Apple is known for its innovative products like the iPhone.
______________________________________________________________________
Source Node 2/4
Node ID: vec3
Similarity: 0.55420566
Text: Apple Inc. has revolutionized the tech industry with its sleek
designs and user-friendly interfaces.
______________________________________________________________________
Source Node 3/4
Node ID: vec4
Similarity: 0.3172258
Text: An apple a day keeps the doctor away, as the saying goes.
______________________________________________________________________
Source Node 4/4
Node ID: vec0
Similarity: 0.25139993
Text: Apple is a popular fruit known for its sweetness and crisp
texture.

```
