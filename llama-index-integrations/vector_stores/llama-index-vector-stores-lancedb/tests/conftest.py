from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex
import lance  # noqa: F401
import pytest
import pytest_asyncio


try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from lancedb.rerankers import LinearCombinationReranker

    deps = True
except ImportError:
    deps = None


@pytest.mark.skipif(
    deps is None,
    reason="Need to install lancedb and huggingface locally to run this test.",
)
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
