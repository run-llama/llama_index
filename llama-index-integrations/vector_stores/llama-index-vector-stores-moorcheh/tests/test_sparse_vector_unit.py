"""Offline unit tests for the sparse-vector paths of MoorchehVectorStore."""

import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import TextNode
from llama_index.vector_stores.moorcheh import MoorchehVectorStore
from llama_index.vector_stores.moorcheh.utils import (
    DefaultMoorchehSparseEmbedding,
    get_default_tokenizer,
)


def _fake_tokenizer(texts: List[str]) -> List[List[int]]:
    return [[1, 1, 2] for _ in texts]


@pytest.fixture()
def client() -> Any:
    fake = MagicMock()
    fake.list_namespaces.return_value = {"namespaces": []}
    with patch(
        "llama_index.vector_stores.moorcheh.base.MoorchehClient", return_value=fake
    ):
        yield fake


def _store(namespace_type: str, **kwargs: Any) -> MoorchehVectorStore:
    return MoorchehVectorStore(
        api_key="key",
        namespace="ns",
        namespace_type=namespace_type,
        vector_dimension=2 if namespace_type == "vector" else None,
        **kwargs,
    )


def test_default_sparse_model_is_built_from_tokenizer(client: Any) -> None:
    store = _store("vector", add_sparse_vector=True, tokenizer=_fake_tokenizer)
    assert isinstance(store._sparse_embedding_model, DefaultMoorchehSparseEmbedding)


def test_add_vector_nodes_attaches_sparse_values(client: Any) -> None:
    store = _store("vector", add_sparse_vector=True, tokenizer=_fake_tokenizer)
    nodes = [
        TextNode(id_="a", text="a", embedding=[0.1, 0.2]),
        TextNode(id_="b", text="b", embedding=[0.3, 0.4]),
    ]
    assert store.add(nodes) == ["a", "b"]

    uploaded: List[Dict[str, Any]] = client.upload_vectors.call_args.kwargs["vectors"]
    assert [v["id"] for v in uploaded] == ["a", "b"]
    for vector in uploaded:
        assert vector["sparse_values"] == {"indices": [1, 2], "values": [2.0, 1.0]}


def test_add_text_nodes_attaches_sparse_values(client: Any) -> None:
    store = _store("text", add_sparse_vector=True, tokenizer=_fake_tokenizer)
    store.add([TextNode(id_="a", text="a"), TextNode(id_="b", text="b")])

    uploaded = client.upload_documents.call_args.kwargs["documents"]
    assert all("sparse_values" in doc for doc in uploaded)


def test_sparse_model_runs_once_per_add(client: Any) -> None:
    model = DefaultMoorchehSparseEmbedding(tokenizer=_fake_tokenizer)
    store = _store("text", add_sparse_vector=True, sparse_embedding_model=model)
    nodes = [TextNode(id_=str(i), text=str(i)) for i in range(5)]

    with patch.object(
        DefaultMoorchehSparseEmbedding,
        "get_text_embedding_batch",
        wraps=model.get_text_embedding_batch,
    ) as batch_fn:
        store.add(nodes)

    assert batch_fn.call_count == 1


def test_missing_embedding_warns(client: Any) -> None:
    store = _store("vector")
    nodes = [
        TextNode(id_="a", text="a", embedding=[0.1, 0.2]),
        TextNode(id_="b", text="b"),
    ]
    with pytest.warns(UserWarning, match="has no embedding"):
        store.add(nodes)


def test_default_tokenizer_requires_transformers() -> None:
    with patch.dict(sys.modules, {"transformers": None}):
        with pytest.raises(ImportError, match="transformers"):
            get_default_tokenizer()
