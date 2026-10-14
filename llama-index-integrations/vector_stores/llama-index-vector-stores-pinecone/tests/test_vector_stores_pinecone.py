import os
import pytest
import time
import uuid

from pinecone import FetchResponse, Pinecone, ServerlessSpec, Vector
from types import SimpleNamespace
from typing import List
from unittest.mock import create_autospec

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    FilterCondition,
    FilterOperator,
    VectorStoreQuery,
)
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.pinecone.base import PineconeIndex

MAX_WAIT_TIME = 60
EMBED_DIM = 1536
MOCK_EMBED_DIM = 4
PINECONE_API_KEY = os.environ.get(
    "PINECONE_API_KEY",
    None,
)
should_skip = not all((PINECONE_API_KEY,))


def test_class():
    names_of_base_classes = [b.__name__ for b in PineconeVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_index_type_matches_installed_client():
    index = Pinecone(api_key="dummy").Index(host="http://localhost")
    assert type(index) is PineconeIndex


@pytest.fixture
def mock_index():
    """An index mock bound to the signatures of the installed client."""
    index = create_autospec(PineconeIndex, instance=True)
    index.upsert.return_value = SimpleNamespace(errors=[])
    return index


@pytest.fixture
def mock_store(mock_index):
    return PineconeVectorStore(pinecone_index=mock_index)


def mock_node(node_id: str = "node-1") -> TextNode:
    return TextNode(
        text="Hello, world!",
        id_=node_id,
        metadata={"some_key": 1},
        embedding=[0.1] * MOCK_EMBED_DIM,
    )


def mock_match(node: TextNode, score: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(
        id=node.node_id,
        values=node.embedding,
        metadata=node_to_metadata_dict(node),
        score=score,
    )


def test_add_upserts_vectors_by_keyword(mock_store, mock_index):
    node = mock_node()

    assert mock_store.add([node]) == [node.node_id]

    call = mock_index.upsert.call_args
    # pinecone>=9 takes no positional arguments at all.
    assert call.args == ()
    assert [entry["id"] for entry in call.kwargs["vectors"]] == [node.node_id]


def test_add_accepts_responses_without_batch_errors(mock_store, mock_index):
    # pinecone<9 responses only carry `upserted_count`.
    mock_index.upsert.return_value = SimpleNamespace(upserted_count=1)

    assert mock_store.add([mock_node()]) == ["node-1"]


def test_add_raises_on_partially_failed_batches(mock_store, mock_index):
    # pinecone>=9 collects per-batch failures on the response instead of raising.
    cause = ConnectionError("connection refused")
    mock_index.upsert.return_value = SimpleNamespace(
        upserted_count=0,
        failed_item_count=1,
        total_item_count=1,
        errors=[
            SimpleNamespace(
                batch_index=0, error=cause, error_message="connection refused"
            )
        ],
    )

    with pytest.raises(RuntimeError, match="connection refused") as excinfo:
        mock_store.add([mock_node()])

    assert excinfo.value.__cause__ is cause


def test_delete_nodes_by_filters_does_not_send_ids(mock_store, mock_index):
    mock_store.delete_nodes(
        filters=MetadataFilters(filters=[MetadataFilter(key="some_key", value=1)])
    )

    call = mock_index.delete.call_args
    # Sending an empty `ids` alongside `filter` is rejected by pinecone>=9.
    assert "ids" not in call.kwargs
    assert call.kwargs["filter"] == {"some_key": {"$eq": 1}}


def test_delete_nodes_by_ids_does_not_send_filter(mock_store, mock_index):
    mock_store.delete_nodes(node_ids=["node-1", "node-2"])

    call = mock_index.delete.call_args
    assert call.kwargs["ids"] == ["node-1", "node-2"]
    assert "filter" not in call.kwargs


def test_delete_nodes_rejects_ids_and_filters_together(mock_store, mock_index):
    with pytest.raises(ValueError):
        mock_store.delete_nodes(
            node_ids=["node-1"],
            filters=MetadataFilters(filters=[MetadataFilter(key="some_key", value=1)]),
        )

    mock_index.delete.assert_not_called()


def test_delete_nodes_without_selectors_does_nothing(mock_store, mock_index):
    mock_store.delete_nodes()

    mock_index.delete.assert_not_called()


@pytest.mark.parametrize(
    "pages",
    [
        # pinecone<9 yields lists of ids, pinecone>=9 yields pages of entries.
        [["doc#a", "doc#b"], ["doc#c"]],
        [
            [SimpleNamespace(id="doc#a"), SimpleNamespace(id="doc#b")],
            [SimpleNamespace(id="doc#c")],
        ],
    ],
    ids=["id_pages", "entry_pages"],
)
def test_delete_prefix_fallback_flattens_listed_ids(mock_store, mock_index, pages):
    mock_index.delete.side_effect = [
        RuntimeError("delete by filter unsupported"),
        None,
        None,
    ]
    mock_index.list.return_value = iter(pages)
    mock_index.fetch.side_effect = lambda *, ids, **kwargs: FetchResponse(
        vectors={
            node_id: Vector(
                id=node_id, values=[0.1] * MOCK_EMBED_DIM, metadata={"doc_id": "doc"}
            )
            for node_id in ids
        }
    )

    mock_store.delete("doc")

    assert [call.kwargs["ids"] for call in mock_index.delete.call_args_list[1:]] == [
        ["doc#a", "doc#b"],
        ["doc#c"],
    ]
    mock_index.list.assert_called_once_with(prefix="doc#", namespace=None)


@pytest.mark.parametrize("entry_pages", [False, True], ids=["id_pages", "entry_pages"])
@pytest.mark.parametrize("ref_doc_id", ["doc1", "doc1#revision"])
def test_delete_fallback_preserves_other_documents(mock_index, entry_pages, ref_doc_id):
    store = PineconeVectorStore(pinecone_index=mock_index, namespace="documents")
    document_ids = [ref_doc_id, ref_doc_id + "0", ref_doc_id + "#revision"]
    stored = {}
    for document_id in document_ids:
        node = mock_node()
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=document_id
        )
        store.add([node])
        entry = mock_index.upsert.call_args.kwargs["vectors"][0]
        stored[entry["id"]] = entry

    def delete(*, filter=None, ids=None, **kwargs):
        if filter is not None:
            raise RuntimeError("delete by filter unsupported")
        for node_id in ids:
            stored.pop(node_id)

    def list_pages(*, prefix, **kwargs):
        for node_id in stored.copy():
            if node_id.startswith(prefix):
                yield [SimpleNamespace(id=node_id) if entry_pages else node_id]

    mock_index.delete.side_effect = delete
    mock_index.list.side_effect = list_pages
    mock_index.fetch.side_effect = lambda *, ids, **kwargs: FetchResponse(
        vectors={
            node_id: Vector(
                id=node_id,
                values=stored[node_id]["values"],
                metadata=stored[node_id]["metadata"],
            )
            for node_id in ids
        },
        namespace=kwargs["namespace"],
    )

    store.delete(ref_doc_id, timeout=5)

    assert set(stored) == {f"{document_id}#node-1" for document_id in document_ids[1:]}
    mock_index.list.assert_called_once_with(
        prefix=f"{ref_doc_id}#", namespace="documents"
    )
    for call in mock_index.fetch.call_args_list:
        assert call.kwargs["namespace"] == "documents"
    for call in mock_index.delete.call_args_list:
        assert call.kwargs["namespace"] == "documents"
        assert call.kwargs["timeout"] == 5


def test_delete_by_metadata_does_not_list_or_fetch(mock_store, mock_index):
    mock_store.delete("doc1", timeout=5)

    mock_index.delete.assert_called_once_with(
        filter={"doc_id": {"$eq": "doc1"}}, namespace=None, timeout=5
    )
    mock_index.list.assert_not_called()
    mock_index.fetch.assert_not_called()


@pytest.mark.parametrize("metadata", [None, {}, {"doc_id": "doc10"}])
def test_delete_fallback_requires_matching_document_metadata(
    mock_store, mock_index, metadata
):
    mock_index.delete.side_effect = RuntimeError("delete by filter unsupported")
    mock_index.list.return_value = iter([["doc1#node-1"]])
    mock_index.fetch.return_value = FetchResponse(
        vectors={
            "doc1#node-1": Vector(id="doc1#node-1", values=[0.1], metadata=metadata)
        }
    )

    mock_store.delete("doc1")

    assert mock_index.delete.call_count == 1


def test_delete_fallback_propagates_fetch_errors(mock_store, mock_index):
    mock_index.delete.side_effect = RuntimeError("delete by filter unsupported")
    mock_index.list.return_value = iter([["doc1#node-1"]])
    mock_index.fetch.side_effect = ConnectionError("fetch failed")

    with pytest.raises(ConnectionError, match="fetch failed"):
        mock_store.delete("doc1")

    assert mock_index.delete.call_count == 1


def test_query_matches_installed_client_signature(mock_store, mock_index):
    node = mock_node()
    mock_index.query.return_value = SimpleNamespace(matches=[mock_match(node)])

    result = mock_store.query(
        VectorStoreQuery(
            query_embedding=[0.1] * MOCK_EMBED_DIM,
            similarity_top_k=1,
        )
    )

    assert result.ids == [node.node_id]
    assert mock_index.query.call_args.args == ()


def test_get_nodes_matches_installed_client_signature(mock_store, mock_index):
    node = mock_node()
    mock_index.describe_index_stats.return_value = {"dimension": MOCK_EMBED_DIM}
    mock_index.query.return_value = SimpleNamespace(matches=[mock_match(node)])

    nodes = mock_store.get_nodes(
        filters=MetadataFilters(filters=[MetadataFilter(key="some_key", value=1)])
    )

    assert [n.node_id for n in nodes] == [node.node_id]


def test_clear_matches_installed_client_signature(mock_store, mock_index):
    mock_store.clear()

    assert mock_index.delete.call_args.kwargs["delete_all"] is True


@pytest.fixture
def nodes():
    return [
        TextNode(
            text="Hello, world 1!",
            metadata={"some_key": 1},
            embedding=[0.3] * EMBED_DIM,
        ),
        TextNode(
            text="Hello, world 2!",
            metadata={"some_key": 2},
            embedding=[0.5] * EMBED_DIM,
        ),
        TextNode(
            text="Hello, world 3!",
            metadata={"some_key": "3"},
            embedding=[0.7] * EMBED_DIM,
        ),
    ]


@pytest.fixture
def pinecone_index():
    index_name = f"{uuid.uuid4()}"

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    pc_index = pc.Index(index_name)

    yield pc_index

    pc.delete_index(index_name)


@pytest.fixture
def index_with_nodes(pinecone_index: PineconeIndex, nodes: List[TextNode]):
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
    )

    # Note: not ideal, but pinecone takes a while to index the nodes
    start_time = time.time()
    while True:
        stats = pinecone_index.describe_index_stats()
        if stats["total_vector_count"] != len(nodes):
            if time.time() - start_time > MAX_WAIT_TIME:
                raise Exception("Index not ready after 60 seconds")

            time.sleep(1)
        else:
            break

    return index


@pytest.mark.skipif(
    should_skip, reason="PINECONE_API_KEY and/or PINECONE_INDEX_NAME not set"
)
def test_basic_e2e(index_with_nodes: VectorStoreIndex):
    nodes = index_with_nodes.as_retriever().retrieve("Hello, world 1!")
    assert len(nodes) == 2


@pytest.mark.skipif(
    should_skip, reason="PINECONE_API_KEY and/or PINECONE_INDEX_NAME not set"
)
def test_retrieval_with_filters(index_with_nodes: VectorStoreIndex):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value=1,
                operator=FilterOperator.EQ,
            ),
            MetadataFilter(
                key="some_key",
                value=2,
                operator=FilterOperator.EQ,
            ),
        ],
        condition=FilterCondition.OR,
    )
    nodes = index_with_nodes.as_retriever(filters=filters).retrieve("Hello, world 1!")
    assert len(nodes) == 2

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value=1,
                operator=FilterOperator.GT,
            ),
        ],
    )
    nodes = index_with_nodes.as_retriever(filters=filters).retrieve("Hello, world 1!")
    assert len(nodes) == 1

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value=[1, 2],
                operator=FilterOperator.IN,
            ),
        ],
    )
    nodes = index_with_nodes.as_retriever(filters=filters).retrieve("Hello, world 1!")
    assert len(nodes) == 2

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="some_key",
                value="3",
                operator=FilterOperator.EQ,
            ),
        ],
    )
    nodes = index_with_nodes.as_retriever(filters=filters).retrieve("Hello, world 1!")
    assert len(nodes) == 1
