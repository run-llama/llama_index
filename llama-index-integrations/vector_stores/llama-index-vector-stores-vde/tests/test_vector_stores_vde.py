from types import SimpleNamespace

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, VectorStoreQuery
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.vde import VDEVectorStore

def test_class_name() -> None:
    assert VDEVectorStore.class_name() == "VDEVectorStore"

def test_init_class() -> None:
    vector_store = VDEVectorStore(
        address="localhost:50051"
    )
    assert vector_store.address == "localhost:50051"

def test_check_health() -> None:
    vector_store = VDEVectorStore(
        address="localhost:50051"
    )
    version, uptime = vector_store._client.health_check()
    assert version is not None
    assert uptime is not None

def test_open_close_collection() -> None:
    vector_store = VDEVectorStore(
        address="localhost:50051",
        collection_name="test_collection",
    )
    assert vector_store._client.has_collection(vector_store.collection_name)
