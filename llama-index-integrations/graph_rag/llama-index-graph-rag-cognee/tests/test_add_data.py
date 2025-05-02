from llama_index.core import Document
import asyncio
import pytest
from llama_index.graph_rag.cognee import CogneeGraphRAG


@pytest.mark.asyncio()
async def test_add_data(monkeypatch):
    # Instantiate cognee GraphRAG
    cogneeGraphRAG = CogneeGraphRAG(
        llm_api_key="",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        graph_db_provider="networkx",
        vector_db_provider="lancedb",
        relational_db_provider="sqlite",
        relational_db_name="cognee_db",
    )

    async def mock_add_return(add, dataset_name):
        return True

    import cognee

    monkeypatch.setattr(cognee, "add", mock_add_return)

    # Gather documents to add to GraphRAG
    documents = [
        Document(
            text="Jessica Miller, Experienced Sales Manager with a strong track record in driving sales growth and building high-performing teams."
        ),
        Document(
            text="David Thompson, Creative Graphic Designer with over 8 years of experience in visual design and branding."
        ),
    ]

    await cogneeGraphRAG.add(documents, "test")
    await cogneeGraphRAG.add(documents[0], "test")


if __name__ == "__main__":
    asyncio.run(test_add_data())
