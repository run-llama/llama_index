import os
import sys
from unittest.mock import patch, MagicMock
from typing import Any

from llama_index.core.llms.mock import MockLLM


@patch("llama_index.vector_stores.qdrant.QdrantVectorStore")
@patch("llama_index.graph_stores.neo4j.Neo4jPropertyGraphStore")
def test_init(mock_neo4j: Any, mock_qdrant: Any):
    os.environ["IS_TESTING"] = "1"

    mock_qdrant.return_value = MagicMock()
    mock_neo4j.return_value = MagicMock()

    print("\n🚀 Starting Pack Initialization Test...")

    modules_to_remove = [
        "llama_index.packs.multimodal_agentic_rag",
        "llama_index.packs.multimodal_agentic_rag.base",
    ]

    for m in modules_to_remove:
        if m in sys.modules:
            del sys.modules[m]

    from llama_index.packs.multimodal_agentic_rag import (
        MultimodalAgenticRAGPack,
    )

    pack = MultimodalAgenticRAGPack(
        llm=MockLLM(),
        embed_model="default",
        qdrant_url="http://localhost:6333",
        neo4j_url="bolt://localhost:7687",
        neo4j_password="fake_password",
        force_recreate=False,
    )

    assert pack is not None
