"""Test recursive queries."""

from typing import Any, Dict, List

from llama_index.core.async_utils import asyncio_run
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.data_structs.data_structs import IndexStruct
from llama_index.core.indices.composability.graph import ComposableGraph
from llama_index.core.indices.keyword_table.simple_base import (
    SimpleKeywordTableIndex,
)
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import Document
from tests.mock_utils.mock_prompts import MOCK_QUERY_KEYWORD_EXTRACT_PROMPT


class MockEmbedding(BaseEmbedding):
    @classmethod
    def class_name(cls) -> str:
        return "MockEmbedding"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        if query == "Foo?":
            return [0, 0, 1, 0, 0]
        elif query == "Orange?":
            return [0, 1, 0, 0, 0]
        elif query == "Cat?":
            return [0, 0, 0, 1, 0]
        else:
            raise ValueError("Invalid query for `_get_query_embedding`.")

    async def _aget_text_embedding(self, text: str) -> List[float]:
        # assume dimensions are 5
        if text == "Hello world.":
            return [1, 0, 0, 0, 0]
        elif text == "This is a test.":
            return [0, 1, 0, 0, 0]
        elif text == "This is another test.":
            return [0, 0, 1, 0, 0]
        elif text == "This is a test v2.":
            return [0, 0, 0, 1, 0]
        elif text == "foo bar":
            return [0, 0, 1, 0, 0]
        elif text == "apple orange":
            return [0, 1, 0, 0, 0]
        elif text == "toronto london":
            return [1, 0, 0, 0, 0]
        elif text == "cat dog":
            return [0, 0, 0, 1, 0]
        else:
            raise ValueError("Invalid text for `mock_get_text_embedding`.")

    def _get_query_embedding(self, query: str) -> List[float]:
        """Mock get query embedding."""
        if query == "Foo?":
            return [0, 0, 1, 0, 0]
        elif query == "Orange?":
            return [0, 1, 0, 0, 0]
        elif query == "Cat?":
            return [0, 0, 0, 1, 0]
        else:
            raise ValueError("Invalid query for `_get_query_embedding`.")

    def _get_text_embedding(self, text: str) -> List[float]:
        """Mock get text embedding."""
        # assume dimensions are 5
        if text == "Hello world.":
            return [1, 0, 0, 0, 0]
        elif text == "This is a test.":
            return [0, 1, 0, 0, 0]
        elif text == "This is another test.":
            return [0, 0, 1, 0, 0]
        elif text == "This is a test v2.":
            return [0, 0, 0, 1, 0]
        elif text == "foo bar":
            return [0, 0, 1, 0, 0]
        elif text == "apple orange":
            return [0, 1, 0, 0, 0]
        elif text == "toronto london":
            return [1, 0, 0, 0, 0]
        elif text == "cat dog":
            return [0, 0, 0, 1, 0]
        else:
            raise ValueError("Invalid text for `mock_get_text_embedding`.")


def test_recursive_query_vector_table(
    documents: List[Document],
    index_kwargs: Dict,
    patch_token_text_splitter,
    patch_llm_predictor,
) -> None:
    """Test query."""
    vector_kwargs = index_kwargs["vector"]
    table_kwargs = index_kwargs["table"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    # try building a list for every two, then a tree
    vector1 = VectorStoreIndex.from_documents(
        documents[0:2], embed_model=MockEmbedding(), **vector_kwargs
    )
    vector2 = VectorStoreIndex.from_documents(
        documents[2:4], embed_model=MockEmbedding(), **vector_kwargs
    )
    list3 = VectorStoreIndex.from_documents(
        documents[4:6], embed_model=MockEmbedding(), **vector_kwargs
    )
    list4 = VectorStoreIndex.from_documents(
        documents[6:8], embed_model=MockEmbedding(), **vector_kwargs
    )
    indices = [vector1, vector2, list3, list4]

    summaries = [
        "foo bar",
        "apple orange",
        "toronto london",
        "cat dog",
    ]

    graph = ComposableGraph.from_indices(
        SimpleKeywordTableIndex,
        indices,
        index_summaries=summaries,
        **table_kwargs,
    )

    custom_query_engines = {
        index.index_id: index.as_query_engine(similarity_top_k=1) for index in indices
    }
    custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
        similarity_top_k=1
    )

    query_str = "Foo?"
    query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)
    response = query_engine.query(query_str)
    assert str(response) == ("Foo?:Foo?:This is another test.")
    query_str = "Orange?"
    response = query_engine.query(query_str)
    assert str(response) == ("Orange?:Orange?:This is a test.")
    query_str = "Cat?"
    response = query_engine.query(query_str)
    assert str(response) == ("Cat?:Cat?:This is a test v2.")


def test_recursive_query_vector_table_query_configs(
    documents: List[Document],
    index_kwargs: Dict,
    patch_llm_predictor,
    patch_token_text_splitter,
) -> None:
    """
    Test query.

    Difference with above test is we specify query config params and
    assert that they're passed in.

    """
    vector_kwargs = index_kwargs["vector"]
    table_kwargs = index_kwargs["table"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    # try building a list for every two, then a tree
    vector1 = VectorStoreIndex.from_documents(
        documents[0:2], embed_model=MockEmbedding(), **vector_kwargs
    )
    vector2 = VectorStoreIndex.from_documents(
        documents[2:4], embed_model=MockEmbedding(), **vector_kwargs
    )
    assert isinstance(vector1.index_struct, IndexStruct)
    assert isinstance(vector2.index_struct, IndexStruct)
    vector1.index_struct.index_id = "vector1"
    vector2.index_struct.index_id = "vector2"
    summaries = [
        "foo bar",
        "apple orange",
    ]

    graph = ComposableGraph.from_indices(
        SimpleKeywordTableIndex,
        [vector1, vector2],
        index_summaries=summaries,
        **table_kwargs,
    )
    assert isinstance(graph, ComposableGraph)

    custom_query_engines = {
        "keyword_table": graph.root_index.as_query_engine(
            query_keyword_extract_template=MOCK_QUERY_KEYWORD_EXTRACT_PROMPT
        ),
        "vector1": vector1.as_query_engine(similarity_top_k=2),
        "vector2": vector2.as_query_engine(similarity_top_k=2),
    }

    query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)
    response = query_engine.query("Foo?")  # type: ignore
    assert str(response) == ("Foo?:Foo?:This is another test.:This is a test v2.")

    response = query_engine.query("Orange?")  # type: ignore
    assert str(response) == ("Orange?:Orange?:This is a test.:Hello world.")


def test_recursive_query_vector_table_async(
    allow_networking: Any,
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    index_kwargs: Dict,
) -> None:
    """Test async query of table index over vector indices."""
    vector_kwargs = index_kwargs["vector"]
    table_kwargs = index_kwargs["table"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    # try building a list for every two, then a tree
    vector1 = VectorStoreIndex.from_documents(
        documents[0:2], embed_model=MockEmbedding(), **vector_kwargs
    )
    vector2 = VectorStoreIndex.from_documents(
        documents[2:4], embed_model=MockEmbedding(), **vector_kwargs
    )
    list3 = VectorStoreIndex.from_documents(
        documents[4:6], embed_model=MockEmbedding(), **vector_kwargs
    )
    list4 = VectorStoreIndex.from_documents(
        documents[6:8], embed_model=MockEmbedding(), **vector_kwargs
    )
    indices = [vector1, vector2, list3, list4]

    summaries = [
        "foo bar",
        "apple orange",
        "toronto london",
        "cat dog",
    ]

    graph = ComposableGraph.from_indices(
        SimpleKeywordTableIndex,
        children_indices=indices,
        index_summaries=summaries,
        **table_kwargs,
    )

    custom_query_engines = {
        index.index_id: index.as_query_engine(similarity_top_k=1) for index in indices
    }
    custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
        similarity_top_k=1
    )

    query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)
    task = query_engine.aquery("Cat?")
    response = asyncio_run(task)
    assert str(response) == ("Cat?:Cat?:This is a test v2.")


def test_recursive_query_vector_vector(
    documents: List[Document],
    patch_llm_predictor,
    patch_token_text_splitter,
    index_kwargs: Dict,
) -> None:
    """Test query."""
    vector_kwargs = index_kwargs["vector"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    # try building a list for every two, then a tree
    vector1 = VectorStoreIndex.from_documents(
        documents[0:2], embed_model=MockEmbedding(), **vector_kwargs
    )
    vector2 = VectorStoreIndex.from_documents(
        documents[2:4], embed_model=MockEmbedding(), **vector_kwargs
    )
    list3 = VectorStoreIndex.from_documents(
        documents[4:6], embed_model=MockEmbedding(), **vector_kwargs
    )
    list4 = VectorStoreIndex.from_documents(
        documents[6:8], embed_model=MockEmbedding(), **vector_kwargs
    )

    indices = [vector1, vector2, list3, list4]

    summary1 = "foo bar"
    summary2 = "apple orange"
    summary3 = "toronto london"
    summary4 = "cat dog"
    summaries = [summary1, summary2, summary3, summary4]

    graph = ComposableGraph.from_indices(
        VectorStoreIndex,
        children_indices=indices,
        index_summaries=summaries,
        embed_model=MockEmbedding(),
        **vector_kwargs,
    )
    custom_query_engines = {
        index.index_id: index.as_query_engine(similarity_top_k=1) for index in indices
    }
    custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
        similarity_top_k=1
    )

    query_str = "Foo?"
    query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)
    response = query_engine.query(query_str)
    assert str(response) == ("Foo?:Foo?:This is another test.")
    query_str = "Orange?"
    response = query_engine.query(query_str)
    assert str(response) == ("Orange?:Orange?:This is a test.")
    query_str = "Cat?"
    response = query_engine.query(query_str)
    assert str(response) == ("Cat?:Cat?:This is a test v2.")
