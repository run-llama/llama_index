import os
import pandas as pd
import asyncio

import cognee
from llama_index.core import Document
from llama_index.graph_rag.cognee import CogneeGraphRAG


async def test_graph_rag_cognee():
    # Gather documents to add to GraphRAG
    news = pd.read_csv(
        "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
    )[:5]
    news.head()
    documents = [
        Document(text=f"{row['title']}: {row['text']}") for i, row in news.iterrows()
    ]

    # Instantiate cognee GraphRAG
    cogneeRAG = CogneeGraphRAG(
        llm_api_key=os.environ["OPENAI_API_KEY"],
        graph_db_provider="networkx",
        vector_db_provider="lancedb",
        relational_db_provider="sqlite",
        db_name="cognee_db",
    )

    # Add data to cognee
    await cogneeRAG.add(documents, "test")
    # Process data into a knowledge graph
    await cogneeRAG.process_data("test")

    # Answer prompt based on knowledge graph
    search_results = await cogneeRAG.search("person")
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")

    # Search for related nodes
    search_results = await cogneeRAG.get_related_nodes("person")
    print("\n\nRelated nodes are:\n")
    for result in search_results:
        print(f"{result}\n")

    # Clean all data from previous runs
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


if __name__ == "__main__":
    asyncio.run(test_graph_rag_cognee())
