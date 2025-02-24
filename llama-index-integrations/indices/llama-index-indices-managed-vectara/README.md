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

Please note that this usage example is for versions >= 0.4.0 and will not be the same as for earlier versions of Vectara ManagedIndex.

First let's initialize the index with some sample documents.
Make sure to always specify a unique `id_` for every document you add to your index.
If you don't specify this parameter, a random id will be generated and the document will be separately added to your corpus every time you run your code.

```python
import os

os.environ["VECTARA_API_KEY"] = "<YOUR_VECTARA_API_KEY>"
os.environ["VECTARA_CORPUS_KEY"] = "<YOUR_VECTARA_CORPUS_KEY>"

from llama_index.indices.managed.vectara import VectaraIndex
from llama_index.core.schema import Document, MediaResource

docs = [
    Document(
        id_="doc1",
        text_resource=MediaResource(
            text="""
            This is test text for Vectara integration with LlamaIndex.
            Users should love their experience with this integration
            """,
        ),
    ),
    Document(
        id_="doc2",
        text_resource=MediaResource(
            text="""
            The Vectara index integration with LlamaIndex implements Vectara's RAG pipeline.
            It can be used both as a retriever and query engine.
            """,
        ),
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
