from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.indices.property_graph.sub_retrievers.text_to_cypher import (
    TextToCypherRetriever,
)
from llama_index.core.schema import QueryBundle


def test_text_to_cypher_summarize_response_preserves_question() -> None:
    mock_graph_store = MagicMock()
    mock_graph_store.get_schema_str.return_value = "Node properties: ()"
    mock_graph_store.structured_query.return_value = [{"count": 42}]

    mock_llm = MagicMock()
    # First call generates cypher, second call summarizes response
    mock_llm.predict.side_effect = [
        "MATCH (f:Flight) RETURN count(f) AS count",
        "There are 42 flights in total.",
    ]

    retriever = TextToCypherRetriever(
        graph_store=mock_graph_store,
        llm=mock_llm,
        summarize_response=True,
    )

    query_str = "How many flights are there? Answer in Spanish."
    nodes = retriever.retrieve_from_graph(QueryBundle(query_str=query_str))

    assert len(nodes) == 1
    assert nodes[0].text == "There are 42 flights in total."

    # Verify that the summarization call received the user's natural language question
    assert mock_llm.predict.call_count == 2
    summarize_call_kwargs = mock_llm.predict.call_args_list[1].kwargs
    assert summarize_call_kwargs["question"] == query_str
    assert summarize_call_kwargs["query"] == "MATCH (f:Flight) RETURN count(f) AS count"
    assert "count" in summarize_call_kwargs["context"]


@pytest.mark.asyncio
async def test_async_text_to_cypher_summarize_response_preserves_question() -> None:
    mock_graph_store = MagicMock()
    mock_graph_store.aget_schema_str = AsyncMock(return_value="Node properties: ()")
    mock_graph_store.astructured_query = AsyncMock(return_value=[{"count": 42}])

    mock_llm = MagicMock()
    mock_llm.apredict = AsyncMock(
        side_effect=[
            "MATCH (f:Flight) RETURN count(f) AS count",
            "There are 42 flights in total.",
        ]
    )

    retriever = TextToCypherRetriever(
        graph_store=mock_graph_store,
        llm=mock_llm,
        summarize_response=True,
    )

    query_str = "How many flights are there? Answer in Spanish."
    nodes = await retriever.aretrieve_from_graph(QueryBundle(query_str=query_str))

    assert len(nodes) == 1
    assert nodes[0].text == "There are 42 flights in total."

    assert mock_llm.apredict.call_count == 2
    summarize_call_kwargs = mock_llm.apredict.call_args_list[1].kwargs
    assert summarize_call_kwargs["question"] == query_str
    assert summarize_call_kwargs["query"] == "MATCH (f:Flight) RETURN count(f) AS count"
    assert "count" in summarize_call_kwargs["context"]
