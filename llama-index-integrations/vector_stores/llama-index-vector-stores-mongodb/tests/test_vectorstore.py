import os
from time import sleep
import pytest
from typing import List

from llama_index.core.schema import Document, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from .conftest import lock


def test_documents(documents: List[Document]) -> None:
    """Sanity check essay was found and documents loaded."""
    assert len(documents) == 25
    assert isinstance(documents[0], Document)


@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None,
    reason="Requires OPENAI_API_KEY in os.environ",
)
def test_nodes(nodes: List[TextNode]) -> None:
    """Test Ingestion Pipeline transforming documents into nodes with embeddings."""
    assert isinstance(nodes, list)
    assert isinstance(nodes[0], TextNode)


@pytest.mark.skipif(
    os.environ.get("MONGODB_URI") is None or os.environ.get("OPENAI_API_KEY") is None,
    reason="Requires MONGODB_URI and OPENAI_API_KEY in os.environ",
)
def test_vectorstore(
    nodes: List[TextNode], vector_store: MongoDBAtlasVectorSearch
) -> None:
    """Test add, query, delete API of MongoDBAtlasVectorSearch."""
    with lock:
        # 0. Clean up the collection
        vector_store._collection.delete_many({})
        sleep(2)

        # 1. Test add()
        ids = vector_store.add(nodes)
        assert set(ids) == {node.node_id for node in nodes}

        # 2. test query()
        query_str = "What are LLMs useful for?"
        n_similar = 2
        query_embedding = OpenAIEmbedding().get_text_embedding(query_str)
        query = VectorStoreQuery(
            query_str=query_str,
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

        assert all(score > 0.89 for score in query_responses.similarities)
        assert any("LLM" in node.text for node in query_responses.nodes)
        assert all(id_res in ids for id_res in query_responses.ids)

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
