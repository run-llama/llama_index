# LlamaIndex Graph Rag Integration: Cognee

Cognee assists developers in introducing greater predictability and management into their Retrieval-Augmented Generation (RAG) workflows through the use of graph architectures, vector stores, and auto-optimizing pipelines. Displaying information as a graph is the clearest way to grasp the content of your documents. Crucially, graphs allow systematic navigation and extraction of data from documents based on their hierarchy.

For more information, visit [Cognee documentation](https://docs.cognee.ai/)

## Installation

```shell
pip install llama-index-graph-rag-cognee
```

## Usage

```python
import os
import pandas as pd
import asyncio

from llama_index.core import Document
from llama_index.graph_rag.cognee import CogneeGraphRAG


async def example_graph_rag_cognee():
    # Gather documents to add to GraphRAG
    news = pd.read_csv(
        "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
    )[:5]
    news.head()
    documents = [
        Document(text=f"{row['title']}: {row['text']}")
        for i, row in news.iterrows()
    ]

    # Instantiate cognee GraphRAG
    cogneeRAG = CogneeGraphRAG(
        llm_api_key=os.environ["OPENAI_API_KEY"],
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        graph_db_provider="networkx",
        vector_db_provider="lancedb",
        relational_db_provider="sqlite",
        relational_db_name="cognee_db",
    )

    # Add data to cognee
    await cogneeRAG.add(documents, "test")

    # Process data into a knowledge graph
    await cogneeRAG.process_data("test")

    # Answer prompt based on knowledge graph
    search_results = await cogneeRAG.search(
        "Tell me who are the people mentioned?"
    )
    print("\n\nAnswer based on knowledge graph:\n")
    for result in search_results:
        print(f"{result}\n")

    # Answer prompt based on RAG
    search_results = await cogneeRAG.rag_search(
        "Tell me who are the people mentioned?"
    )
    print("\n\nAnswer based on RAG:\n")
    for result in search_results:
        print(f"{result}\n")

    # Search for related nodes in graph
    search_results = await cogneeRAG.get_related_nodes("person")
    print("\n\nRelated nodes are:\n")
    for result in search_results:
        print(f"{result}\n")


if __name__ == "__main__":
    asyncio.run(example_graph_rag_cognee())
```

## Supported databases

**Relational databases:** SQLite, PostgreSQL

**Vector databases:** LanceDB, PGVector, QDrant, Weviate

**Graph databases:** Neo4j, NetworkX
