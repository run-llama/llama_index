import sys
from unittest.mock import MagicMock
from llama_index.data_structs.node import Node
from llama_index.vector_stores.types import NodeEmbeddingResult

from llama_index.vector_stores.weaviate import WeaviateVectorStore


def test_weaviate_add() -> None:
    # mock import
    sys.modules["weaviate"] = MagicMock()
    weaviate_client = MagicMock()
    batch_context_manager = MagicMock()
    weaviate_client.batch.__enter__.return_value = batch_context_manager

    vector_store = WeaviateVectorStore(weaviate_client=weaviate_client)

    vector_store.add(
        [
            NodeEmbeddingResult(
                id="test node id",
                node=Node(text="test node text"),
                embedding=[0.5, 0.5],
                doc_id="test doc id",
            )
        ]
    )

    args, _ = batch_context_manager.add_data_object.call_args
    assert args[-1] == [0.5, 0.5]
