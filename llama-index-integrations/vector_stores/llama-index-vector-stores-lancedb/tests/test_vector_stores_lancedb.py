import tantivy  # noqa
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex
import pytest
import lance  # noqa: F401
import pytest
import pytest_asyncio
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.lancedb import LanceDBVectorStore

try:
    from lancedb.rerankers import LinearCombinationReranker
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    deps = True
except ImportError:
    deps = None


@pytest_asyncio.fixture
async def index() -> VectorStoreIndex:
    vector_store = LanceDBVectorStore(
        overfetch_factor=1,
        mode="overwrite",
        reranker=LinearCombinationReranker(weight=0.3),
    )
    nodes = [
        TextNode(
            text="test1",
            id_="11111111-1111-1111-1111-111111111111",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-0")},
        ),
        TextNode(
            text="test2",
            id_="22222222-2222-2222-2222-222222222222",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-1")},
        ),
        TextNode(
            text="test3",
            id_="33333333-3333-3333-3333-333333333333",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test-2")},
        ),
    ]
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    vector_store.add(nodes=nodes)

    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


def test_class():
    names_of_base_classes = [b.__name__ for b in LanceDBVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb and huggingface locally to run this test.",
)
def test_vector_query(index: VectorStoreIndex) -> None:
    retriever = index.as_retriever()
    response = retriever.retrieve("test1")
    assert response[0].id_ == "11111111-1111-1111-1111-111111111111"


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb and huggingface locally to run this test.",
)
def test_fts_query(index: VectorStoreIndex) -> None:
    try:
        response = index.as_retriever(
            vector_store_kwargs={"query_type": "fts"}
        ).retrieve("test")
    except Warning as e:
        pass

    response = index.as_retriever(vector_store_kwargs={"query_type": "fts"}).retrieve(
        "test1"
    )
    assert response[0].id_ == "11111111-1111-1111-1111-111111111111"


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb and huggingface locally to run this test.",
)
def test_hybrid_query(index: VectorStoreIndex) -> None:
    response = index.as_retriever(
        vector_store_kwargs={"query_type": "hybrid"}
    ).retrieve("test")

    assert response[0].id_ == "11111111-1111-1111-1111-111111111111"


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb and huggingface locally to run this test.",
)
def test_delete(index: VectorStoreIndex) -> None:
    index.delete(doc_id="test-0")
    assert index.vector_store._table.count_rows() == 2
