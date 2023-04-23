"""Test recursive queries."""

import asyncio
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from gpt_index.data_structs.data_structs_v2 import V2IndexStruct
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.composability.graph import ComposableGraph
from gpt_index.indices.keyword_table.simple_base import GPTSimpleKeywordTableIndex
from gpt_index.indices.vector_store.vector_indices import (
    GPTPineconeIndex,
    GPTSimpleVectorIndex,
)
from gpt_index.langchain_helpers.chain_wrapper import (
    LLMPredictor,
)
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.readers.schema.base import Document
from tests.indices.vector_store.utils import MockPineconeIndex
from tests.mock_utils.mock_predict import (
    mock_llmpredictor_predict,
)
from tests.mock_utils.mock_prompts import (
    MOCK_QUERY_KEYWORD_EXTRACT_PROMPT,
)
from tests.mock_utils.mock_text_splitter import mock_token_splitter_newline
from tests.mock_utils.mock_utils import mock_tokenizer


def mock_get_query_embedding(query: str) -> List[float]:
    """Mock get query embedding."""
    if query == "Foo?":
        return [0, 0, 1, 0, 0]
    elif query == "Orange?":
        return [0, 1, 0, 0, 0]
    elif query == "Cat?":
        return [0, 0, 0, 1, 0]
    else:
        raise ValueError("Invalid query for `mock_get_query_embedding`.")


def mock_get_text_embedding(text: str) -> List[float]:
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


def mock_get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Mock get text embeddings."""
    return [mock_get_text_embedding(text) for text in texts]


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "get_query_embedding", side_effect=mock_get_query_embedding
)
def test_recursive_query_vector_table(
    _mock_query_embed: Any,
    _mock_get_text_embeds: Any,
    _mock_get_text_embed: Any,
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_predict: Any,
    _mock_split_text: Any,
    documents: List[Document],
    index_kwargs: Dict,
) -> None:
    """Test query."""
    vector_kwargs = index_kwargs["vector"]
    table_kwargs = index_kwargs["table"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    # try building a list for every two, then a tree
    vector1 = GPTSimpleVectorIndex.from_documents(documents[0:2], **vector_kwargs)
    vector2 = GPTSimpleVectorIndex.from_documents(documents[2:4], **vector_kwargs)
    list3 = GPTSimpleVectorIndex.from_documents(documents[4:6], **vector_kwargs)
    list4 = GPTSimpleVectorIndex.from_documents(documents[6:8], **vector_kwargs)

    summaries = [
        "foo bar",
        "apple orange",
        "toronto london",
        "cat dog",
    ]

    graph = ComposableGraph.from_indices(
        GPTSimpleKeywordTableIndex,
        [vector1, vector2, list3, list4],
        index_summaries=summaries,
        **table_kwargs
    )
    assert isinstance(graph, ComposableGraph)
    query_str = "Foo?"
    query_engine = graph.as_query_engine()
    response = query_engine.query(query_str)
    assert str(response) == ("Foo?:Foo?:This is another test.")
    query_str = "Orange?"
    response = query_engine.query(query_str)
    assert str(response) == ("Orange?:Orange?:This is a test.")
    query_str = "Cat?"
    response = query_engine.query(query_str)
    assert str(response) == ("Cat?:Cat?:This is a test v2.")

    # test serialize and then back
    # use composable graph struct
    with TemporaryDirectory() as tmpdir:
        graph.save_to_disk(str(Path(tmpdir) / "tmp.json"))
        graph = ComposableGraph.load_from_disk(str(Path(tmpdir) / "tmp.json"))
        query_engine = graph.as_query_engine()
        response = query_engine.query(query_str)
        assert str(response) == ("Cat?:Cat?:This is a test v2.")


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "get_query_embedding", side_effect=mock_get_query_embedding
)
def test_recursive_query_vector_table_query_configs(
    _mock_query_embed: Any,
    _mock_get_text_embeds: Any,
    _mock_get_text_embed: Any,
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_predict: Any,
    _mock_split_text: Any,
    documents: List[Document],
    index_kwargs: Dict,
) -> None:
    """Test query.

    Difference with above test is we specify query config params and
    assert that they're passed in.

    """
    vector_kwargs = index_kwargs["vector"]
    table_kwargs = index_kwargs["table"]
    # try building a tre for a group of 4, then a list
    # use a diff set of documents
    # try building a list for every two, then a tree
    vector1 = GPTSimpleVectorIndex.from_documents(documents[0:2], **vector_kwargs)
    vector2 = GPTSimpleVectorIndex.from_documents(documents[2:4], **vector_kwargs)
    assert isinstance(vector1.index_struct, V2IndexStruct)
    assert isinstance(vector2.index_struct, V2IndexStruct)
    vector1.index_struct.index_id = "vector1"
    vector2.index_struct.index_id = "vector2"
    summaries = [
        "foo bar",
        "apple orange",
    ]

    graph = ComposableGraph.from_indices(
        GPTSimpleKeywordTableIndex,
        [vector1, vector2],
        index_summaries=summaries,
        **table_kwargs
    )
    assert isinstance(graph, ComposableGraph)

    custom_retrievers = {
        "keyword_table": graph.root_index.as_retriever(
            query_keyword_extract_template=MOCK_QUERY_KEYWORD_EXTRACT_PROMPT
        ),
        "vector1": vector1.as_retriever(similarity_top_k=2),
        "vector2": vector2.as_retriever(similarity_top_k=2),
    }

    query_engine = graph.as_query_engine(custom_retrievers=custom_retrievers)
    response = query_engine.query("Foo?")  # type: ignore
    assert str(response) == ("Foo?:Foo?:This is another test.:This is a test v2.")

    response = query_engine.query("Orange?")  # type: ignore
    assert str(response) == ("Orange?:Orange?:This is a test.:Hello world.")

    # test serialize and then back
    # use composable graph struct
    with TemporaryDirectory() as tmpdir:
        graph.save_to_disk(str(Path(tmpdir) / "tmp.json"))
        graph = ComposableGraph.load_from_disk(str(Path(tmpdir) / "tmp.json"))
        # cast to Any to avoid mypy error
        query_engine = graph.as_query_engine(custom_retrievers=custom_retrievers)
        response = query_engine.query("Orange?")
        assert str(response) == ("Orange?:Orange?:This is a test.:Hello world.")


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "get_query_embedding", side_effect=mock_get_query_embedding
)
def test_recursive_query_vector_table_async(
    _mock_query_embed: Any,
    _mock_get_text_embeds: Any,
    _mock_get_text_embed: Any,
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_predict: Any,
    _mock_split_text: Any,
    documents: List[Document],
    index_kwargs: Dict,
) -> None:
    """Test async query of table index over vector indices."""
    vector_kwargs = index_kwargs["vector"]
    table_kwargs = index_kwargs["table"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    # try building a list for every two, then a tree
    vector1 = GPTSimpleVectorIndex.from_documents(documents[0:2], **vector_kwargs)
    vector2 = GPTSimpleVectorIndex.from_documents(documents[2:4], **vector_kwargs)
    list3 = GPTSimpleVectorIndex.from_documents(documents[4:6], **vector_kwargs)
    list4 = GPTSimpleVectorIndex.from_documents(documents[6:8], **vector_kwargs)

    summaries = [
        "foo bar",
        "apple orange",
        "toronto london",
        "cat dog",
    ]

    graph = ComposableGraph.from_indices(
        GPTSimpleKeywordTableIndex,
        [vector1, vector2, list3, list4],
        index_summaries=summaries,
        **table_kwargs
    )
    assert isinstance(graph, ComposableGraph)

    query_engine = graph.as_query_engine()
    task = query_engine.aquery("Cat?")
    response = asyncio.run(task)
    assert str(response) == ("Cat?:Cat?:This is a test v2.")


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "get_query_embedding", side_effect=mock_get_query_embedding
)
def test_recursive_query_vector_vector(
    _mock_query_embed: Any,
    _mock_get_text_embeds: Any,
    _mock_get_text_embed: Any,
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_predict: Any,
    _mock_split_text: Any,
    documents: List[Document],
    index_kwargs: Dict,
    retriever_kwargs: Dict,
) -> None:
    """Test query."""
    vector_kwargs = index_kwargs["vector"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    # try building a list for every two, then a tree
    vector1 = GPTSimpleVectorIndex.from_documents(documents[0:2], **vector_kwargs)
    vector2 = GPTSimpleVectorIndex.from_documents(documents[2:4], **vector_kwargs)
    list3 = GPTSimpleVectorIndex.from_documents(documents[4:6], **vector_kwargs)
    list4 = GPTSimpleVectorIndex.from_documents(documents[6:8], **vector_kwargs)

    summary1 = "foo bar"
    summary2 = "apple orange"
    summary3 = "toronto london"
    summary4 = "cat dog"
    summaries = [summary1, summary2, summary3, summary4]

    graph = ComposableGraph.from_indices(
        GPTSimpleVectorIndex,
        [vector1, vector2, list3, list4],
        index_summaries=summaries,
        **vector_kwargs
    )
    query_str = "Foo?"
    query_engine = graph.as_query_engine()
    response = query_engine.query(query_str)
    assert str(response) == ("Foo?:Foo?:This is another test.")
    query_str = "Orange?"
    response = query_engine.query(query_str)
    assert str(response) == ("Orange?:Orange?:This is a test.")
    query_str = "Cat?"
    response = query_engine.query(query_str)
    assert str(response) == ("Cat?:Cat?:This is a test v2.")

    # test serialize and then back
    # use composable graph struct
    with TemporaryDirectory() as tmpdir:
        graph.save_to_disk(str(Path(tmpdir) / "tmp.json"))
        graph = ComposableGraph.load_from_disk(str(Path(tmpdir) / "tmp.json"))
        query_engine = graph.as_query_engine()
        response = query_engine.query(query_str)
        assert str(response) == ("Cat?:Cat?:This is a test v2.")


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
@patch.object(LLMPredictor, "total_tokens_used", return_value=0)
@patch.object(LLMPredictor, "__init__", return_value=None)
@patch.object(
    OpenAIEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding
)
@patch.object(
    OpenAIEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
@patch.object(
    OpenAIEmbedding, "get_query_embedding", side_effect=mock_get_query_embedding
)
def test_recursive_query_pinecone_pinecone(
    _mock_query_embed: Any,
    _mock_get_text_embeds: Any,
    _mock_get_text_embed: Any,
    _mock_init: Any,
    _mock_total_tokens_used: Any,
    _mock_predict: Any,
    _mock_split_text: Any,
    documents: List[Document],
    index_kwargs: Dict,
    retriever_kwargs: Dict,
) -> None:
    """Test composing pinecone index on top of pinecone index."""
    # NOTE: mock pinecone import
    sys.modules["pinecone"] = MagicMock()
    # NOTE: mock pinecone index, use separate instances
    #       to make testing easier
    pinecone_index1 = MockPineconeIndex()
    pinecone_index2 = MockPineconeIndex()
    pinecone_index3 = MockPineconeIndex()
    pinecone_index4 = MockPineconeIndex()
    pinecone_index5 = MockPineconeIndex()

    pinecone_kwargs = index_kwargs["pinecone"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    # try building a list for every two, then a tree
    pinecone1 = GPTPineconeIndex.from_documents(
        documents[0:2],
        pinecone_index=pinecone_index1,
        tokenizer=mock_tokenizer,
        **pinecone_kwargs
    )
    pinecone2 = GPTPineconeIndex.from_documents(
        documents[2:4],
        pinecone_index=pinecone_index2,
        tokenizer=mock_tokenizer,
        **pinecone_kwargs
    )
    pinecone3 = GPTPineconeIndex.from_documents(
        documents[4:6],
        pinecone_index=pinecone_index3,
        tokenizer=mock_tokenizer,
        **pinecone_kwargs
    )
    pinecone4 = GPTPineconeIndex.from_documents(
        documents[6:8],
        pinecone_index=pinecone_index4,
        tokenizer=mock_tokenizer,
        **pinecone_kwargs
    )

    summary1 = "foo bar"
    summary2 = "apple orange"
    summary3 = "toronto london"
    summary4 = "cat dog"
    summaries = [summary1, summary2, summary3, summary4]

    graph = ComposableGraph.from_indices(
        GPTPineconeIndex,
        [pinecone1, pinecone2, pinecone3, pinecone4],
        index_summaries=summaries,
        pinecone_index=pinecone_index5,
        tokenizer=mock_tokenizer,
        **pinecone_kwargs
    )
    query_str = "Foo?"
    query_engine = graph.as_query_engine()
    response = query_engine.query(query_str)
    assert str(response) == ("Foo?:Foo?:This is another test.")
    query_str = "Orange?"
    response = query_engine.query(query_str)
    assert str(response) == ("Orange?:Orange?:This is a test.")
    query_str = "Cat?"
    response = query_engine.query(query_str)
    assert str(response) == ("Cat?:Cat?:This is a test v2.")

    # test serialize and then back
    # use composable graph struct
    with TemporaryDirectory() as tmpdir:
        graph.save_to_disk(str(Path(tmpdir) / "tmp.json"))
        index_kwargs = {
            index_id: {
                "pinecone_index": pinecone_index,
                "tokenizer": mock_tokenizer,
            }
            for index_id, pinecone_index in zip(
                [
                    pinecone1.index_struct.index_id,
                    pinecone2.index_struct.index_id,
                    pinecone3.index_struct.index_id,
                    pinecone4.index_struct.index_id,
                    graph.root_id,
                ],
                [
                    pinecone_index1,
                    pinecone_index2,
                    pinecone_index3,
                    pinecone_index4,
                    pinecone_index5,
                ],
            )
        }
        graph = ComposableGraph.load_from_disk(
            str(Path(tmpdir) / "tmp.json"), index_kwargs=index_kwargs
        )
        query_engine = graph.as_query_engine()
        response = query_engine.query(query_str)
        assert str(response) == ("Cat?:Cat?:This is a test v2.")
