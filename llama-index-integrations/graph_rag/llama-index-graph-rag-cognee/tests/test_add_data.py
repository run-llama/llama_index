import pandas as pd
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
        db_name="cognee_db",
    )

    # Mock logging to graphistry
    async def mock_add_return(add, dataset_name):
        return True

    import cognee

    monkeypatch.setattr(cognee, "add", mock_add_return)

    # Gather documents to add to GraphRAG
    news = pd.read_csv(
        "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
    )[:5]
    news.head()
    documents = [
        Document(text=f"{row['title']}: {row['text']}") for i, row in news.iterrows()
    ]

    await cogneeGraphRAG.add(documents, "test")
    await cogneeGraphRAG.add(documents[0], "test")


if __name__ == "__main__":
    asyncio.run(test_add_data())
