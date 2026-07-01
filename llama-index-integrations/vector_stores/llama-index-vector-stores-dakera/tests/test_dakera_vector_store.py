"""
Unit tests for DakeraVectorStore.

All HTTP calls are intercepted with ``respx`` (an httpx mock library), so no
live Dakera server is required.
"""

import json
from typing import Any, Dict
from unittest.mock import patch

import pytest
import respx
from httpx import Response

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
)
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.dakera import DakeraVectorStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:3300"
AGENT_ID = "test-agent"
API_KEY = "test-key"


@pytest.fixture()
def store() -> DakeraVectorStore:
    return DakeraVectorStore(
        base_url=BASE_URL,
        agent_id=AGENT_ID,
        api_key=API_KEY,
        top_k=5,
    )


@pytest.fixture()
def sample_node() -> TextNode:
    return TextNode(
        text="The quick brown fox jumps over the lazy dog.",
        id_="node-abc-123",
        metadata={"source": "test", "importance": 0.8},
    )


def _make_memory(memory_id: str, content: str, node: TextNode) -> Dict[str, Any]:
    """Build a fake Dakera memory record matching the API shape."""
    metadata = node_to_metadata_dict(node)
    meta_tag = f"_llama_meta={json.dumps(metadata)}"
    return {
        "id": memory_id,
        "content": content,
        "agent_id": AGENT_ID,
        "tags": [f"node_id={node.node_id}", meta_tag],
    }


# ---------------------------------------------------------------------------
# Class-shape tests (no network required)
# ---------------------------------------------------------------------------


def test_inherits_base_pydantic_vector_store():
    names = [b.__name__ for b in DakeraVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names


def test_class_name(store: DakeraVectorStore):
    assert store.class_name() == "DakeraVectorStore"


def test_stores_text_flag(store: DakeraVectorStore):
    assert store.stores_text is True


def test_is_embedding_query_false(store: DakeraVectorStore):
    """Dakera embeds server-side — LlamaIndex must pass query_str, not vectors."""
    assert store.is_embedding_query is False


def test_client_property(store: DakeraVectorStore):
    import httpx
    assert isinstance(store.client, httpx.Client)


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------


@respx.mock
def test_add_single_node(store: DakeraVectorStore, sample_node: TextNode):
    memory_id = "mem-001"
    respx.post(f"{BASE_URL}/v1/memory/store").mock(
        return_value=Response(
            200,
            json={"memory": {"id": memory_id, "content": sample_node.text}},
        )
    )

    ids = store.add([sample_node])

    assert ids == [memory_id]
    assert respx.calls.call_count == 1

    sent = json.loads(respx.calls[0].request.content)
    assert sent["agent_id"] == AGENT_ID
    assert sent["content"] == sample_node.text
    assert sent["importance"] == pytest.approx(0.8)
    assert any("node_id=node-abc-123" in t for t in sent["tags"])
    assert any("_llama_meta=" in t for t in sent["tags"])


@respx.mock
def test_add_multiple_nodes(store: DakeraVectorStore):
    nodes = [
        TextNode(text=f"content {i}", id_=f"node-{i}")
        for i in range(3)
    ]
    for i, node in enumerate(nodes):
        respx.post(f"{BASE_URL}/v1/memory/store").mock(
            return_value=Response(
                200, json={"memory": {"id": f"mem-{i}", "content": node.text}}
            )
        )

    ids = store.add(nodes)
    assert ids == ["mem-0", "mem-1", "mem-2"]
    assert respx.calls.call_count == 3


@respx.mock
def test_add_raises_on_http_error(store: DakeraVectorStore, sample_node: TextNode):
    respx.post(f"{BASE_URL}/v1/memory/store").mock(
        return_value=Response(500, json={"error": "internal server error"})
    )

    with pytest.raises(Exception):
        store.add([sample_node])


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_looks_up_then_forgets(store: DakeraVectorStore, sample_node: TextNode):
    memory = _make_memory("mem-001", sample_node.text, sample_node)

    # Search returns one hit with the exact node_id tag
    respx.post(f"{BASE_URL}/v1/memory/search").mock(
        return_value=Response(
            200, json={"memories": [{"memory": memory, "score": 0.9}]}
        )
    )
    respx.post(f"{BASE_URL}/v1/memory/forget").mock(
        return_value=Response(200, json={"deleted": 1})
    )

    store.delete(sample_node.node_id)

    assert respx.calls.call_count == 2
    forget_body = json.loads(respx.calls[1].request.content)
    assert forget_body["memory_ids"] == ["mem-001"]
    assert forget_body["agent_id"] == AGENT_ID


@respx.mock
def test_delete_with_explicit_memory_ids(store: DakeraVectorStore):
    respx.post(f"{BASE_URL}/v1/memory/forget").mock(
        return_value=Response(200, json={"deleted": 2})
    )

    store.delete("any-ref", memory_ids=["mem-A", "mem-B"])

    # Should skip the search step
    assert respx.calls.call_count == 1
    body = json.loads(respx.calls[0].request.content)
    assert sorted(body["memory_ids"]) == ["mem-A", "mem-B"]


@respx.mock
def test_delete_no_op_when_nothing_found(store: DakeraVectorStore):
    respx.post(f"{BASE_URL}/v1/memory/search").mock(
        return_value=Response(200, json={"memories": []})
    )

    store.delete("node-does-not-exist")

    # Only the search call; no forget call
    assert respx.calls.call_count == 1


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------


@respx.mock
def test_query_returns_nodes_scores_ids(store: DakeraVectorStore, sample_node: TextNode):
    memory = _make_memory("mem-001", sample_node.text, sample_node)
    respx.post(f"{BASE_URL}/v1/memory/search").mock(
        return_value=Response(
            200,
            json={"memories": [{"memory": memory, "score": 0.95}]},
        )
    )

    result = store.query(VectorStoreQuery(query_str="quick brown fox", similarity_top_k=5))

    assert result.ids == ["mem-001"]
    assert result.similarities == pytest.approx([0.95])
    assert len(result.nodes) == 1
    assert result.nodes[0].get_content() == sample_node.text

    sent = json.loads(respx.calls[0].request.content)
    assert sent["query"] == "quick brown fox"
    assert sent["top_k"] == 5
    assert sent["agent_id"] == AGENT_ID


@respx.mock
def test_query_uses_store_top_k_as_default(store: DakeraVectorStore):
    respx.post(f"{BASE_URL}/v1/memory/search").mock(
        return_value=Response(200, json={"memories": []})
    )

    # similarity_top_k defaults to 1 in VectorStoreQuery; but store.top_k=5
    # When similarity_top_k > 0, it wins over the store default.
    store.query(VectorStoreQuery(query_str="test"))
    sent = json.loads(respx.calls[0].request.content)
    # similarity_top_k=1 (dataclass default) takes precedence
    assert sent["top_k"] == 1


def test_query_raises_without_query_str(store: DakeraVectorStore):
    with pytest.raises(ValueError, match="query_str"):
        store.query(VectorStoreQuery(query_str=None))


@respx.mock
def test_query_fallback_textnode_when_no_meta_tag(store: DakeraVectorStore):
    """If a memory has no _llama_meta tag, fall back to a plain TextNode."""
    memory = {
        "id": "mem-999",
        "content": "bare content",
        "agent_id": AGENT_ID,
        "tags": [],
    }
    respx.post(f"{BASE_URL}/v1/memory/search").mock(
        return_value=Response(
            200, json={"memories": [{"memory": memory, "score": 0.5}]}
        )
    )

    result = store.query(VectorStoreQuery(query_str="anything"))

    from llama_index.core.schema import TextNode as TN
    assert isinstance(result.nodes[0], TN)
    assert result.nodes[0].text == "bare content"
    assert result.nodes[0].node_id == "mem-999"


# ---------------------------------------------------------------------------
# Session-scoped store
# ---------------------------------------------------------------------------


@respx.mock
def test_session_id_propagated_to_store(sample_node: TextNode):
    store = DakeraVectorStore(
        base_url=BASE_URL,
        agent_id=AGENT_ID,
        session_id="sess-xyz",
    )
    respx.post(f"{BASE_URL}/v1/memory/store").mock(
        return_value=Response(
            200, json={"memory": {"id": "mem-s1", "content": ""}}
        )
    )

    store.add([sample_node])
    sent = json.loads(respx.calls[0].request.content)
    assert sent["session_id"] == "sess-xyz"


# ---------------------------------------------------------------------------
# Authorization header
# ---------------------------------------------------------------------------


@respx.mock
def test_auth_header_sent(store: DakeraVectorStore, sample_node: TextNode):
    respx.post(f"{BASE_URL}/v1/memory/store").mock(
        return_value=Response(
            200, json={"memory": {"id": "m1", "content": ""}}
        )
    )

    store.add([sample_node])
    auth = respx.calls[0].request.headers.get("authorization")
    assert auth == f"Bearer {API_KEY}"


@respx.mock
def test_no_auth_header_when_no_key(sample_node: TextNode):
    store = DakeraVectorStore(base_url=BASE_URL, agent_id=AGENT_ID)
    respx.post(f"{BASE_URL}/v1/memory/store").mock(
        return_value=Response(
            200, json={"memory": {"id": "m2", "content": ""}}
        )
    )

    store.add([sample_node])
    auth = respx.calls[0].request.headers.get("authorization")
    assert auth is None
