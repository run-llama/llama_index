from typing import Any, List

import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore, FaissMapVectorStore
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

EMBEDDING_DIMENSION = 1536


def test_class():
    names_of_base_classes = [b.__name__ for b in FaissVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes
    names_of_base_classes = [b.__name__ for b in FaissMapVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def get_index() -> Any:
    """Get a faiss index object."""
    import faiss

    return faiss.IndexFlatL2(EMBEDDING_DIMENSION)


def get_map_index() -> Any:
    """Get a faiss map index object."""
    import faiss

    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    return faiss.IndexIDMap2(faiss_index)


def text_to_embedding(text: str) -> List[float]:
    """Convert text to a unique embedding using ASCII values."""
    ascii_values = [float(ord(char)) for char in text]
    return ascii_values[:EMBEDDING_DIMENSION] + [0.0] * (
        EMBEDDING_DIMENSION - len(ascii_values)
    )


@pytest.fixture(scope="session")
def node_embeddings() -> list[TextNode]:
    """Return a list of TextNodes with embeddings."""
    return [
        TextNode(
            text="foo",
            id_="12c70eed-5779-4008-aba0-596e003f6443",
            metadata={
                "genre": "Mystery",
                "pages": 10,
            },
            embedding=text_to_embedding("foo"),
        ),
        TextNode(
            text="bar",
            id_="f7d81cb3-bb42-47e6-96f5-17db6860cd11",
            metadata={
                "genre": "Comedy",
                "pages": 5,
            },
            embedding=text_to_embedding("bar"),
        ),
        TextNode(
            text="baz",
            id_="469e9537-7bc5-4669-9ff6-baa0ed086236",
            metadata={
                "genre": "Thriller",
                "pages": 20,
            },
            embedding=text_to_embedding("baz"),
        ),
    ]


class TestFaissMapVectorStore:
    def test_add_documents(self, node_embeddings: List[TextNode]) -> None:
        """Test adding documents to faiss map vector store."""
        vector_store = FaissMapVectorStore(faiss_index=get_map_index())

        # Add nodes to the faiss map vector
        vector_store.add(node_embeddings)

        query = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
        )
        result = vector_store.query(query)
        assert result.similarities[0] == 0.0
        assert result.ids[0] == "469e9537-7bc5-4669-9ff6-baa0ed086236"

        assert len(vector_store._faiss_id_to_node_id_map) == 3
        assert len(vector_store._node_id_to_faiss_id_map) == 3
        assert (
            vector_store._faiss_id_to_node_id_map[0]
            == "12c70eed-5779-4008-aba0-596e003f6443"
        )
        assert (
            vector_store._faiss_id_to_node_id_map[1]
            == "f7d81cb3-bb42-47e6-96f5-17db6860cd11"
        )
        assert (
            vector_store._faiss_id_to_node_id_map[2]
            == "469e9537-7bc5-4669-9ff6-baa0ed086236"
        )
        assert (
            vector_store._node_id_to_faiss_id_map[
                "12c70eed-5779-4008-aba0-596e003f6443"
            ]
            == 0
        )
        assert (
            vector_store._node_id_to_faiss_id_map[
                "f7d81cb3-bb42-47e6-96f5-17db6860cd11"
            ]
            == 1
        )
        assert (
            vector_store._node_id_to_faiss_id_map[
                "469e9537-7bc5-4669-9ff6-baa0ed086236"
            ]
            == 2
        )

    def test_delete_nodes(self, node_embeddings: List[TextNode]) -> None:
        """Test deleting nodes from faiss map vector store."""
        vector_store = FaissMapVectorStore(faiss_index=get_map_index())

        # Add nodes to the faiss map vector
        vector_store.add(node_embeddings)

        query = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
        )
        result = vector_store.query(query)
        assert result.similarities[0] == 0.0
        assert result.ids[0] == "469e9537-7bc5-4669-9ff6-baa0ed086236"

        vector_store.delete_nodes(
            node_ids=["469e9537-7bc5-4669-9ff6-baa0ed086236"],
        )

        query = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
        )
        result = vector_store.query(query)
        print(result)
        assert result.similarities[0] == 64.0
        assert result.ids[0] == "f7d81cb3-bb42-47e6-96f5-17db6860cd11"

        assert len(vector_store._faiss_id_to_node_id_map) == 2
        assert len(vector_store._node_id_to_faiss_id_map) == 2
        assert (
            vector_store._faiss_id_to_node_id_map[0]
            == "12c70eed-5779-4008-aba0-596e003f6443"
        )
        assert (
            vector_store._faiss_id_to_node_id_map[1]
            == "f7d81cb3-bb42-47e6-96f5-17db6860cd11"
        )
        assert (
            vector_store._node_id_to_faiss_id_map[
                "12c70eed-5779-4008-aba0-596e003f6443"
            ]
            == 0
        )
        assert (
            vector_store._node_id_to_faiss_id_map[
                "f7d81cb3-bb42-47e6-96f5-17db6860cd11"
            ]
            == 1
        )

    def test_delete(self, node_embeddings: List[TextNode]) -> None:
        """Test deleting nodes from faiss map vector store."""
        vector_store = FaissMapVectorStore(faiss_index=get_map_index())

        # Add nodes to the faiss map vector
        vector_store.add(node_embeddings)

        query = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
        )
        result = vector_store.query(query)
        assert result.similarities[0] == 0.0
        assert result.ids[0] == "469e9537-7bc5-4669-9ff6-baa0ed086236"

        vector_store.delete(
            ref_doc_id="469e9537-7bc5-4669-9ff6-baa0ed086236",
        )

        query = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
        )
        result = vector_store.query(query)
        print(result)
        assert result.similarities[0] == 64.0
        assert result.ids[0] == "f7d81cb3-bb42-47e6-96f5-17db6860cd11"

        assert len(vector_store._faiss_id_to_node_id_map) == 2
        assert len(vector_store._node_id_to_faiss_id_map) == 2
        assert (
            vector_store._faiss_id_to_node_id_map[0]
            == "12c70eed-5779-4008-aba0-596e003f6443"
        )
        assert (
            vector_store._faiss_id_to_node_id_map[1]
            == "f7d81cb3-bb42-47e6-96f5-17db6860cd11"
        )
        assert (
            vector_store._node_id_to_faiss_id_map[
                "12c70eed-5779-4008-aba0-596e003f6443"
            ]
            == 0
        )
        assert (
            vector_store._node_id_to_faiss_id_map[
                "f7d81cb3-bb42-47e6-96f5-17db6860cd11"
            ]
            == 1
        )

    def test_delete_nodes_with_filters(self, node_embeddings: List[TextNode]) -> None:
        """Test deleting nodes from faiss map vector store with filters."""
        vector_store = FaissMapVectorStore(faiss_index=get_map_index())

        # Add nodes to the faiss map vector
        vector_store.add(node_embeddings)

        query = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
        )
        result = vector_store.query(query)
        assert result.similarities[0] == 0.0
        assert result.ids[0] == "469e9537-7bc5-4669-9ff6-baa0ed086236"

        try:
            vector_store.delete_nodes(
                filters=MetadataFilters(
                    filters=[
                        MetadataFilter(key="genre", value="Thriller", operator="=="),
                        MetadataFilter(key="pages", value=10, operator=">"),
                    ]
                ),
            )
            raise AssertionError
        except NotImplementedError:
            # Metadata filters not implemented for Faiss yet.
            pass

    def test_persist_and_load(self, node_embeddings: List[TextNode]) -> None:
        """Test persisting and loading from faiss map vector store."""
        vector_store = FaissMapVectorStore(faiss_index=get_map_index())

        # Add nodes to the faiss map vector
        vector_store.add(node_embeddings)

        query = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
        )
        result = vector_store.query(query)
        assert result.similarities[0] == 0.0
        assert result.ids[0] == "469e9537-7bc5-4669-9ff6-baa0ed086236"

        # Persist the vector store
        vector_store.persist()

        # Load the vector store
        loaded_vector_store = FaissMapVectorStore.from_persist_dir()

        assert (
            loaded_vector_store._node_id_to_faiss_id_map
            == vector_store._node_id_to_faiss_id_map
        )
        assert (
            loaded_vector_store._faiss_id_to_node_id_map
            == vector_store._faiss_id_to_node_id_map
        )

        # query from loaded vector store
        query = VectorStoreQuery(
            query_embedding=text_to_embedding("baz"),
            similarity_top_k=1,
        )
        result = loaded_vector_store.query(query)
        assert result.similarities[0] == 0.0
        assert result.ids[0] == "469e9537-7bc5-4669-9ff6-baa0ed086236"

        # delete from original vector store
        vector_store.delete(
            ref_doc_id="469e9537-7bc5-4669-9ff6-baa0ed086236",
        )

        assert (
            loaded_vector_store._node_id_to_faiss_id_map
            != vector_store._node_id_to_faiss_id_map
        )
        assert (
            loaded_vector_store._faiss_id_to_node_id_map
            != vector_store._faiss_id_to_node_id_map
        )

        # query from loaded vector store without delete
        result = loaded_vector_store.query(query)
        assert result.similarities[0] == 0.0
        assert result.ids[0] == "469e9537-7bc5-4669-9ff6-baa0ed086236"
