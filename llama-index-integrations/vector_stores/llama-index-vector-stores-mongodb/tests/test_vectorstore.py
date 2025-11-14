from time import sleep
from typing import List

from llama_index.core.schema import Document, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch, index

from .conftest import lock


def test_documents(documents: List[Document]) -> None:
    """Sanity check essay was found and documents loaded."""
    assert len(documents) == 25
    assert isinstance(documents[0], Document)


def test_nodes(nodes: List[TextNode]) -> None:
    """Test Ingestion Pipeline transforming documents into nodes with embeddings."""
    assert isinstance(nodes, list)
    assert isinstance(nodes[0], TextNode)


def test_vectorstore(
    nodes: List[TextNode],
    vector_store: MongoDBAtlasVectorSearch,
    embed_model: OpenAIEmbedding,
) -> None:
    """Test add, query, delete API of MongoDBAtlasVectorSearch."""
    with lock:
        # 0. Clean up the collection
        vector_store._collection.delete_many({})
        sleep(2)

        # 1. Test add()
        ids = vector_store.add(nodes)
        assert set(ids) == {node.node_id for node in nodes}
        # Post-ingest stabilization (avoid transient zero-result queries right after insert)
        for _ in range(5):
            if vector_store._collection.count_documents({}) >= len(ids):
                break
            sleep(1)
        sleep(2)

        # 2a. test query(): default (vector search)
        query_str = "What are LLMs useful for?"
        n_similar = 2
        query_embedding = embed_model.get_text_embedding(query_str)
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=n_similar,
        )
        result_found = False
        query_responses = None
        retries = 5
        while retries and not result_found:
            query_responses = vector_store.query(query=query)
            if len(query_responses.nodes) == n_similar:
                result_found = True
            else:
                sleep(2)
                retries -= 1

        assert all(score > 0.75 for score in query_responses.similarities)
        assert any("LLM" in node.text for node in query_responses.nodes)
        assert all(id_res in ids for id_res in query_responses.ids)

        # 2b. test query() default with simple filter

        # In order to filter within $vectorSearch,
        # one needs to have an index on the field.
        # One can do this by adding an additional member to "fields" list of vector index
        # like so: { "type": "filter", "path": "text }
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="text",
                    value="How do we best augment LLMs with our own private data?",
                    operator=FilterOperator.NE,
                )
            ]
        )
        query = VectorStoreQuery(
            query_str=query_str,
            query_embedding=query_embedding,
            similarity_top_k=n_similar,
            filters=filters,
        )
        responses_with_filter = vector_store.query(query=query)
        assert len(responses_with_filter.ids) == n_similar
        assert all(
            filters.filters[0].value not in node.text
            for node in responses_with_filter.nodes
        )

        # 2c. test query() default with compound filters
        filter_out_texts = [
            "How do we best augment LLMs with our own private data?",
            "easily used with LLMs.",
        ]
        filters_compound = MetadataFilters(
            condition=FilterCondition.AND,
            filters=[
                MetadataFilter(
                    key="text", value=filter_out_texts[0], operator=FilterOperator.NE
                ),
                MetadataFilter(
                    key="text", value=filter_out_texts[1], operator=FilterOperator.NE
                ),
            ],
        )
        query = VectorStoreQuery(
            query_str=query_str,
            query_embedding=query_embedding,
            similarity_top_k=n_similar,
            filters=filters_compound,
        )
        responses_with_filter_compound = vector_store.query(query=query)
        assert len(responses_with_filter_compound.ids) == n_similar
        assert all(
            ftext not in node.text
            for node in responses_with_filter_compound.nodes
            for ftext in filter_out_texts
        )
        assert set(responses_with_filter_compound.ids) != set(responses_with_filter.ids)

        # 2d. test query() full-text search
        #   - no embedding

        query = VectorStoreQuery(
            query_str="llamaindex",
            similarity_top_k=4,
            mode=VectorStoreQueryMode.TEXT_SEARCH,
        )
        result_found = False
        retries = 5
        while retries and not result_found:
            fulltext_result = vector_store.query(query=query)
            if fulltext_result.ids:  # if len(fulltext_result.nodes) == n_similar:
                result_found = True
            else:
                sleep(2)
                retries -= 1
        assert len(fulltext_result.ids) == 3
        assert all("LlamaIndex" in node.text for node in fulltext_result.nodes)

        # 2e. test query() hybrid search
        n_similar = 10
        query = VectorStoreQuery(
            query_str="llamaindex",
            query_embedding=query_embedding,  # "What are LLMs useful for?"
            similarity_top_k=n_similar,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.5,
        )
        hybrid_result = vector_store.query(query=query)
        assert len(hybrid_result.ids) == n_similar
        assert not all("LlamaIndex" in node.text for node in hybrid_result.nodes[:3])
        assert not all("LLM" in node.text for node in hybrid_result.nodes[:3])

        # 3. Test delete()
        # Remember, the current API deletes by *ref_doc_id*, not *node_id*.
        # In our case, we began with only one document,
        # so deleting the ref_doc_id from any node
        # should delete ALL the nodes.
        n_docs = vector_store._collection.count_documents({})
        assert n_docs == len(ids)
        remove_id = query_responses.nodes[0].ref_doc_id
        sleep(2)
        retries = 5
        while retries:
            vector_store.delete(remove_id)
            n_remaining = vector_store._collection.count_documents({})
            if n_remaining == n_docs:
                sleep(2)
                retries -= 1
            else:
                retries = 0
        assert n_remaining == n_docs - 1


# Shared test data setup for DEFAULT mode filter tests
def _setup_default_mode_filter_test_data(vector_store: MongoDBAtlasVectorSearch) -> None:
    """Configure index and insert test data for DEFAULT mode filter tests."""
    collection = vector_store._collection
    vector_index_name = vector_store._vector_index_name

    # Update vector index to include filter paths
    index.update_vector_search_index(
        collection=collection,
        index_name=vector_index_name,
        dimensions=1536,
        path="embedding",
        similarity="cosine",
        filters=["metadata.category", "metadata.priority", "metadata.tags"],
        wait_until_complete=120,
    )

    # Clean and insert test data
    vector_store._collection.delete_many({})
    sleep(1)

    test_nodes = [
        TextNode(
            text="Machine learning fundamentals",
            embedding=[0.9] * 1536,
            metadata={"category": "ai", "priority": "high", "tags": ["ml", "education"]}
        ),
        TextNode(
            text="Deep learning with neural networks",
            embedding=[0.85] * 1536,
            metadata={"category": "ai", "priority": "low", "tags": ["dl", "education"]}
        ),
        TextNode(
            text="Cooking recipes and techniques",
            embedding=[0.1] * 1536,
            metadata={"category": "cooking", "priority": "high", "tags": ["recipes"]}
        ),
        TextNode(
            text="Travel destinations in Europe",
            embedding=[0.05] * 1536,
            metadata={"category": "travel", "priority": "low", "tags": ["europe"]}
        ),
    ]

    ids = vector_store.add(test_nodes)
    assert len(ids) == 4

    # Wait for indexing
    for _ in range(5):
        if vector_store._collection.count_documents({}) >= len(ids):
            break
        sleep(1)
    sleep(7)  # Index rebuild + propagation


def test_default_mode_filter_eq_operator_applies_at_database_level(
    vector_store: MongoDBAtlasVectorSearch,
) -> None:
    """
    Verify EQ filter applied at MongoDB $vectorSearch stage, not post-processed.
    """
    with lock:
        _setup_default_mode_filter_test_data(vector_store)

        query = VectorStoreQuery(
            query_embedding=[0.88] * 1536,
            similarity_top_k=10,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="category", value="ai", operator=FilterOperator.EQ)
                ]
            )
        )

        # Retry for index propagation
        result = None
        for _ in range(10):
            result = vector_store.query(query)
            if len(result.nodes) >= 2:
                break
            sleep(2)

        assert result is not None
        assert len(result.nodes) == 2
        assert all(node.metadata.get("category") == "ai" for node in result.nodes)

        vector_store._collection.delete_many({})


def test_default_mode_filter_in_operator_applies_at_database_level(
    vector_store: MongoDBAtlasVectorSearch,
) -> None:
    """
    Verify IN filter with multiple values applied at MongoDB $vectorSearch stage.
    """
    with lock:
        _setup_default_mode_filter_test_data(vector_store)

        query = VectorStoreQuery(
            query_embedding=[0.88] * 1536,
            similarity_top_k=10,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="tags", value=["ml", "recipes"], operator=FilterOperator.IN)
                ]
            )
        )

        result = vector_store.query(query)

        assert len(result.nodes) == 2
        assert all(
            any(tag in node.metadata.get("tags", []) for tag in ["ml", "recipes"])
            for node in result.nodes
        )

        vector_store._collection.delete_many({})


def test_default_mode_filter_and_condition_applies_at_database_level(
    vector_store: MongoDBAtlasVectorSearch,
) -> None:
    """
    Verify compound AND filter applied at MongoDB $vectorSearch stage.
    """
    with lock:
        _setup_default_mode_filter_test_data(vector_store)

        query = VectorStoreQuery(
            query_embedding=[0.88] * 1536,
            similarity_top_k=10,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="category", value="ai", operator=FilterOperator.EQ),
                    MetadataFilter(key="priority", value="high", operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.AND
            )
        )

        result = vector_store.query(query)

        assert len(result.nodes) == 1
        assert result.nodes[0].metadata.get("category") == "ai"
        assert result.nodes[0].metadata.get("priority") == "high"

        vector_store._collection.delete_many({})


def test_default_mode_filter_or_condition_applies_at_database_level(
    vector_store: MongoDBAtlasVectorSearch,
) -> None:
    """
    Verify compound OR filter applied at MongoDB $vectorSearch stage.
    """
    with lock:
        _setup_default_mode_filter_test_data(vector_store)

        query = VectorStoreQuery(
            query_embedding=[0.88] * 1536,
            similarity_top_k=10,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="category", value="cooking", operator=FilterOperator.EQ),
                    MetadataFilter(key="priority", value="high", operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR
            )
        )

        # Retry for index propagation
        result = None
        for _ in range(10):
            result = vector_store.query(query)
            if len(result.nodes) >= 2:
                break
            sleep(2)

        assert result is not None
        assert len(result.nodes) == 2

        vector_store._collection.delete_many({})
