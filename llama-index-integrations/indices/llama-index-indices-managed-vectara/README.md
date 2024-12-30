# LlamaIndex Managed Integration: Vectara

The Vectara Index provides a simple implementation to Vectara's end-to-end RAG pipeline,
including data ingestion, document retrieval, reranking results, summary generation, and hallucination evaluation.

## Setup

First, make sure you have the latest LlamaIndex version installed.

Next, install the Vectara Index:

```
pip install -U llama-index-indices-managed-vectara
```

Finally, set up your Vectara corpus. If you don't have a Vectara account, you can [sign up](https://vectara.com/integrations/llamaindex) and follow our [Quick Start](https://docs.vectara.com/docs/quickstart) guide to create a corpus and an API key (make sure it has both indexing and query permissions).

## Usage

First let's initialize the index with some sample documents.

```python
import os

os.environ["VECTARA_API_KEY"] = "<YOUR_VECTARA_API_KEY>"
os.environ["VECTARA_CORPUS_ID"] = "<YOUR_VECTARA_CORPUS_ID>"
os.environ["VECTARA_CUSTOMER_ID"] = "<YOUR_VECTARA_CUSTOMER_ID>"

from llama_index.indices.managed.vectara import VectaraIndex
from llama_index.core.schema import Document

docs = [
    Document(
        text="""
        This is test text for Vectara integration with LlamaIndex.
        Users should love their experience with this integration
        """,
    ),
    Document(
        text="""
        The Vectara index integration with LlamaIndex implements Vectara's RAG pipeline.
        It can be used both as a retriever and query engine.
        """,
    ),
]

index = VectaraIndex.from_documents(docs)
```

You can now use this index to retrieve documents.

```python
# Retrieves the top search result
retriever = index.as_retriever(similarity_top_k=1)

results = retriever.retrieve("How will users feel about this new tool?")
print(results[0])
```

You can also use it as a query engine to get a generated summary from the retrieved results.

```python
query_engine = index.as_query_engine()

results = query_engine.query(
    "Which company has partnered with Vectara to implement their RAG pipeline as an index?"
)
print(f"Generated summary: {results.response}\n")
print("Top sources:")
for node in results.source_nodes[:2]:
    print(node)
```

If you want to see the full features and capabilities of `VectaraIndex`, check out this Jupyter [notebook](https://github.com/vectara/example-notebooks/blob/main/notebooks/using-vectara-with-llamaindex.ipynb).
