import os
import pytest

from llama_index.core import Document, KnowledgeGraphIndex, StorageContext
from llama_index.core.data_structs.struct_type import IndexStructType
from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.mongodb.base import MongoDBKVStore

from .conftest import lock


@pytest.mark.skipif(
    os.environ.get("MONGODB_URI") is None or os.environ.get("OPENAI_API_KEY") is None,
    reason="Requires MONGODB_URI and OPENAI_API_KEY in os.environ",
)
def test_knowledge_graph_index_storage() -> None:
    """Test MongoDBKVStore handling index structs as part of KnowledgeGraphIndex."""
    with lock:
        index_store = KVIndexStore(
            kvstore=MongoDBKVStore.from_uri(
                uri=os.environ.get("MONGODB_URI"),
                db_name=os.environ.get("MONGODB_DATABASE"),
            ),
            namespace=os.environ.get("MONGODB_COLLECTION"),
        )

        storage_context = StorageContext.from_defaults(
            index_store=index_store,
        )

        coll = index_store._collection
        index_store._kvstore._db[coll].delete_many({})

        # Create knowledge graph index
        documents = [Document.example() for _ in range(5)]
        for _ in range(2):
            KnowledgeGraphIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                max_triplets_per_chunk=10,
                include_embeddings=True,
            )

        # Test getting all KG index structs
        index_structs = index_store.index_structs()
        assert len(index_structs) == 2
        struct_one, struct_two = index_structs
        assert struct_one.get_type() == IndexStructType.KG
        assert struct_two.get_type() == IndexStructType.KG

        # Test getting specific KG index structs
        assert struct_one == index_store.get_index_struct(struct_one.index_id)
        assert struct_two == index_store.get_index_struct(struct_two.index_id)

        # Test deleting KG index structs
        index_store.delete_index_struct(struct_one.index_id)
        assert len(index_store.index_structs()) == 1
        assert struct_two == index_store.get_index_struct()
