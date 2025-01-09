import os
import asyncio

import cognee
import pytest
from llama_index.core import Document
from llama_index.graph_rag.cognee import CogneeGraphRAG


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not available to test Cognee integration",
)
@pytest.mark.asyncio()
async def test_graph_rag_cognee():
    documents = [
        Document(
            text="Jessica Miller, Experienced Sales Manager with a strong track record in driving sales growth and building high-performing teams."
        ),
        Document(
            text="David Thompson, Creative Graphic Designer with over 8 years of experience in visual design and branding."
        ),
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
    search_results = await cogneeRAG.search("Tell me who are the people mentioned?")

    assert len(search_results) > 0, "No search results found"

    print("\n\nAnswer based on knowledge graph:\n")
    for result in search_results:
        print(f"{result}\n")

    # Answer prompt based on RAG
    search_results = await cogneeRAG.rag_search("Tell me who are the people mentioned?")

    assert len(search_results) > 0, "No search results found"

    print("\n\nAnswer based on RAG:\n")
    for result in search_results:
        print(f"{result}\n")

    # Search for related nodes
    search_results = await cogneeRAG.get_related_nodes("person")
    print("\n\nRelated nodes are:\n")
    for result in search_results:
        print(f"{result}\n")

    assert len(search_results) > 0, "No search results found"

    # Clean all data from previous runs
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


if __name__ == "__main__":
    asyncio.run(test_graph_rag_cognee())
