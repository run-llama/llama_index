"""Vector store."""

from gpt_index.vector_stores.simple import get_top_k_nodes
from gpt_index.vector_stores.types import VectorStoreQueryConfig
from gpt_index.data_structs.node_v2 import Node
from unittest.mock import patch
from typing import Dict, cast
import datetime


def test_get_top_k_nodes() -> None:
    """Test get top k nodes."""

    times = [0, 1, 2, 3]
    key = "__last_accessed__"
    nodes = [
        Node(text="lorem ipsum", node_info={key: time}, doc_id=str(idx))
        for idx, time in enumerate(times)
    ]

    # try with high time decay
    query_embedding = [0.0, 0.0, 1.0]
    node_embeddings = [[0.0, 0.0, 1.0] for _ in range(4)]

    query_config = VectorStoreQueryConfig(
        use_time_decay=True, time_decay_rate=0.9999, similarity_top_k=1
    )

    similarities, ids = get_top_k_nodes(
        query_embedding, nodes, node_embeddings, query_config=query_config, now=4.0
    )
    assert len(similarities) == 1
    assert ids == ["3"]

    assert cast(Dict, nodes[0].node_info)[key] == 0
    assert cast(Dict, nodes[1].node_info)[key] == 1
    assert cast(Dict, nodes[2].node_info)[key] == 2
    assert cast(Dict, nodes[3].node_info)[key] != 3

    # try with low time decay - all nodes should be weighted similarly
    # mock datetime.now
    times = [0, 1, 2, 3]
    key = "__last_accessed__"
    nodes = [
        Node(text="lorem ipsum", node_info={key: time}, doc_id=str(idx))
        for idx, time in enumerate(times)
    ]
    query_config = VectorStoreQueryConfig(
        use_time_decay=True, time_decay_rate=0.000000000002, similarity_top_k=1
    )
    query_embedding = [0.0, 0.0, 1.0]
    # downweight newer nodes (so that newer nodes are less relevant)
    node_embeddings = [[0.0, 0.0, -float(idx)] for idx in range(4)]
    similarities, ids = get_top_k_nodes(
        query_embedding, nodes, node_embeddings, query_config=query_config, now=4.0
    )
    assert len(similarities) == 1
    assert ids == ["0"]
    assert cast(Dict, nodes[0].node_info)[key] != 0
    assert cast(Dict, nodes[3].node_info)[key] == 3
