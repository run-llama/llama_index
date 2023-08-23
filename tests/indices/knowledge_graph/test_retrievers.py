from typing import Any, List
from unittest.mock import patch

from llama_index.graph_stores import SimpleGraphStore
from llama_index.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.indices.knowledge_graph.retrievers import KGTableRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.schema import Document
from llama_index.storage.storage_context import StorageContext
from tests.indices.knowledge_graph.test_base import MockEmbedding, mock_extract_triplets
from tests.mock_utils.mock_prompts import MOCK_QUERY_KEYWORD_EXTRACT_PROMPT


@patch.object(
    KnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_as_retriever(
    _patch_extract_triplets: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test query."""
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    index = KnowledgeGraphIndex.from_documents(
        documents, service_context=mock_service_context, storage_context=storage_context
    )
    retriever: KGTableRetriever = index.as_retriever()  # type: ignore
    nodes = retriever.retrieve(QueryBundle("foo"))
    # when include_text is True, the first node is the raw text
    # the second node is the query
    rel_initial_text = (
        f"The following are knowledge sequence in max depth"
        f" {retriever.graph_store_query_depth} "
        f"in the form of directed graph like:\n"
        f"`subject -[predicate]->, object, <-[predicate_next_hop]-, object_next_hop ...`"
    )

    raw_text = "['foo', 'is', 'bar']"
    query = rel_initial_text + "\n" + raw_text
    assert len(nodes) == 2
    assert nodes[1].node.get_content() == query


@patch.object(
    KnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_retrievers(
    _patch_extract_triplets: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    # test specific retriever class
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex.from_documents(
        documents, service_context=mock_service_context, storage_context=storage_context
    )
    retriever = KGTableRetriever(
        index,
        query_keyword_extract_template=MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
        graph_store=graph_store,
    )
    query_bundle = QueryBundle(query_str="foo", custom_embedding_strs=["foo"])
    nodes = retriever.retrieve(query_bundle)
    assert (
        nodes[1].node.get_content()
        == "The following are knowledge sequence in max depth 2"
        " in the form of directed graph like:\n"
        "`subject -[predicate]->, object, <-[predicate_next_hop]-, object_next_hop ...`"
        "\n['foo', 'is', 'bar']"
    )


@patch.object(
    KnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_retriever_no_text(
    _patch_extract_triplets: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    # test specific retriever class
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex.from_documents(
        documents, service_context=mock_service_context, storage_context=storage_context
    )
    retriever = KGTableRetriever(
        index,
        query_keyword_extract_template=MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
        include_text=False,
        graph_store=graph_store,
    )
    query_bundle = QueryBundle(query_str="foo", custom_embedding_strs=["foo"])
    nodes = retriever.retrieve(query_bundle)
    assert (
        nodes[0].node.get_content()
        == "The following are knowledge sequence in max depth 2"
        " in the form of directed graph like:\n"
        "`subject -[predicate]->, object, <-[predicate_next_hop]-, object_next_hop ...`"
        "\n['foo', 'is', 'bar']"
    )


@patch.object(
    KnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_retrieve_similarity(
    _patch_extract_triplets: Any,
    documents: List[Document],
    mock_service_context: ServiceContext,
) -> None:
    """Test query."""
    mock_service_context.embed_model = MockEmbedding()
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex.from_documents(
        documents,
        include_embeddings=True,
        service_context=mock_service_context,
        storage_context=storage_context,
    )
    retriever = KGTableRetriever(index, similarity_top_k=2, graph_store=graph_store)

    # returns only two rel texts to use for generating response
    # uses hyrbid query by default
    nodes = retriever.retrieve(QueryBundle("foo"))
    assert (
        nodes[1].node.get_content()
        == "The following are knowledge sequence in max depth 2"
        " in the form of directed graph like:\n"
        "`subject -[predicate]->, object, <-[predicate_next_hop]-, object_next_hop ...`"
        "\n['foo', 'is', 'bar']"
    )
