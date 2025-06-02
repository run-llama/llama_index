import sys

import pytest
from llama_index.graph_rag.cognee import CogneeGraphRAG


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="mock strategy requires python3.10 or higher"
)
@pytest.mark.asyncio
async def test_get_graph_url(monkeypatch):
    # Instantiate cognee GraphRAG
    cogneeRAG = CogneeGraphRAG(
        llm_api_key="",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        graph_db_provider="networkx",
        vector_db_provider="lancedb",
        relational_db_provider="sqlite",
        relational_db_name="cognee_db",
    )

    # Mock logging to graphistry
    def mock_graphistry_return(username, password):
        return True

    import graphistry

    monkeypatch.setattr(graphistry, "login", mock_graphistry_return)

    # Mock render of graph
    async def mock_render_return(graph):
        return "link"

    from cognee.shared import utils

    monkeypatch.setattr(utils, "render_graph", mock_render_return)

    await cogneeRAG.get_graph_url("password", "username")

    from cognee.base_config import get_base_config

    assert get_base_config().graphistry_password == "password", (
        "Password was not set properly"
    )
    assert get_base_config().graphistry_username == "username", (
        "Username was not set properly"
    )
