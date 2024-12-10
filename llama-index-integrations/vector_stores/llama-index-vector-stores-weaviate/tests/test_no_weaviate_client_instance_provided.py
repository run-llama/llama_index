import weaviate
import weaviate.embedded
from llama_index.core.schema import (
    TextNode,
)
from llama_index.vector_stores.weaviate import (
    WeaviateVectorStore,
)
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)


# This method was moved to its own module to not conflict with the module-level client() fixture in test_vector_stores_weaviate_sync.py.
# (trying to open two parallel connections to embedded Weaviate leads to a port conflict)
def test_no_weaviate_client_instance_provided():
    """Tests that the creation of a Weaviate client within the WeaviateVectorStore constructor works."""
    vector_store = WeaviateVectorStore(
        client_kwargs={"embedded_options": weaviate.embedded.EmbeddedOptions()}
    )

    # Make sure that the vector store is functional by calling some basic methods
    vector_store.add([TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3])])
    query = VectorStoreQuery(
        query_embedding=[0.3, 0.0, 0.0],
        similarity_top_k=10,
        query_str="test",
        mode=VectorStoreQueryMode.DEFAULT,
    )
    results = vector_store.query(query)
    assert len(results.nodes) == 1
    weaviate_client = vector_store.client
    del vector_store
    assert (
        not weaviate_client.is_connected()
    )  # As the Weaviate client was created within WeaviateVectorStore, it lies in its responsibility to close the connection when it is not longer needed
