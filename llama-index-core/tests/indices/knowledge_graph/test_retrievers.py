from typing import Any, List, Tuple
from unittest.mock import patch

from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.core.indices.knowledge_graph.retrievers import KGTableRetriever
from llama_index.core.schema import Document, QueryBundle
from llama_index.core.storage.storage_context import StorageContext
from tests.mock_utils.mock_prompts import MOCK_QUERY_KEYWORD_EXTRACT_PROMPT


class MockEmbedding(BaseEmbedding):
    @classmethod
    def class_name(cls) -> str:
        return "MockEmbedding"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        del query
        return [0, 0, 1, 0, 0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        # assume dimensions are 4
        if text == "('foo', 'is', 'bar')":
            return [1, 0, 0, 0]
        elif text == "('hello', 'is not', 'world')":
            return [0, 1, 0, 0]
        elif text == "('Jane', 'is mother of', 'Bob')":
            return [0, 0, 1, 0]
        elif text == "foo":
            return [0, 0, 0, 1]
        else:
            raise ValueError("Invalid text for `mock_get_text_embedding`.")

    def _get_text_embedding(self, text: str) -> List[float]:
        """Mock get text embedding."""
        # assume dimensions are 4
        if text == "('foo', 'is', 'bar')":
            return [1, 0, 0, 0]
        elif text == "('hello', 'is not', 'world')":
            return [0, 1, 0, 0]
        elif text == "('Jane', 'is mother of', 'Bob')":
            return [0, 0, 1, 0]
        elif text == "foo":
            return [0, 0, 0, 1]
        else:
            raise ValueError("Invalid text for `mock_get_text_embedding`.")

    def _get_query_embedding(self, query: str) -> List[float]:
        """Mock get query embedding."""
        del query
        return [0, 0, 1, 0, 0]


def mock_extract_triplets(text: str) -> List[Tuple[str, str, str]]:
    """Mock extract triplets."""
    lines = text.split("\n")
    triplets: List[Tuple[str, str, str]] = []
    for line in lines:
        tokens = line[1:-1].split(",")
        tokens = [t.strip() for t in tokens]

        subj, pred, obj = tokens
        triplets.append((subj, pred, obj))
    return triplets


@patch.object(
    KnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_as_retriever(_patch_extract_triplets: Any, documents: List[Document]) -> None:
    """Test query."""
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    index = KnowledgeGraphIndex.from_documents(
        documents, storage_context=storage_context
    )
    retriever: KGTableRetriever = index.as_retriever()  # type: ignore
    nodes = retriever.retrieve(QueryBundle("foo"))
    # when include_text is True, the first node is the raw text
    # the second node is the query
    rel_initial_text = (
        f"The following are knowledge sequence in max depth"
        f" {retriever.graph_store_query_depth} "
        f"in the form of directed graph like:\n"
        f"`subject -[predicate]->, object, <-[predicate_next_hop]-,"
        f" object_next_hop ...`"
    )

    raw_text = "['foo', 'is', 'bar']"
    query = rel_initial_text + "\n" + raw_text
    assert len(nodes) == 2
    assert nodes[1].node.get_content() == query


@patch.object(
    KnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_retrievers(_patch_extract_triplets: Any, documents: List[Document]) -> None:
    # test specific retriever class
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex.from_documents(
        documents, storage_context=storage_context
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
        "`subject -[predicate]->, object, <-[predicate_next_hop]-,"
        " object_next_hop ...`"
        "\n['foo', 'is', 'bar']"
    )


@patch.object(
    KnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_retriever_no_text(
    _patch_extract_triplets: Any, documents: List[Document]
) -> None:
    # test specific retriever class
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex.from_documents(
        documents, storage_context=storage_context
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
        "`subject -[predicate]->, object, <-[predicate_next_hop]-,"
        " object_next_hop ...`"
        "\n['foo', 'is', 'bar']"
    )


@patch.object(
    KnowledgeGraphIndex, "_extract_triplets", side_effect=mock_extract_triplets
)
def test_retrieve_similarity(
    _patch_extract_triplets: Any, documents: List[Document]
) -> None:
    """Test query."""
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex.from_documents(
        documents,
        include_embeddings=True,
        storage_context=storage_context,
        embed_model=MockEmbedding(),
    )
    retriever = KGTableRetriever(index, similarity_top_k=2, graph_store=graph_store)

    # returns only two rel texts to use for generating response
    # uses hyrbid query by default
    nodes = retriever.retrieve(QueryBundle("foo"))
    assert (
        nodes[1].node.get_content()
        == "The following are knowledge sequence in max depth 2"
        " in the form of directed graph like:\n"
        "`subject -[predicate]->, object, <-[predicate_next_hop]-,"
        " object_next_hop ...`"
        "\n['foo', 'is', 'bar']"
    )
