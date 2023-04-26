"""Test recursive queries."""

import asyncio
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from gpt_index.data_structs.data_structs_v2 import V2IndexStruct
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.composability.graph import ComposableGraph
from gpt_index.indices.keyword_table.simple_base import GPTSimpleKeywordTableIndex
from gpt_index.indices.vector_store.base import GPTVectorStoreIndex
from gpt_index.langchain_helpers.chain_wrapper import (
    LLMPredictor,
)
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.readers.schema.base import Document
from gpt_index.storage.storage_context import StorageContext
from gpt_index.vector_stores.pinecone import PineconeVectorStore
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
    vector1 = GPTVectorStoreIndex.from_documents(documents[0:2], **vector_kwargs)
    vector2 = GPTVectorStoreIndex.from_documents(documents[2:4], **vector_kwargs)
    list3 = GPTVectorStoreIndex.from_documents(documents[4:6], **vector_kwargs)
    list4 = GPTVectorStoreIndex.from_documents(documents[6:8], **vector_kwargs)

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
    vector1 = GPTVectorStoreIndex.from_documents(documents[0:2], **vector_kwargs)
    vector2 = GPTVectorStoreIndex.from_documents(documents[2:4], **vector_kwargs)
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
    vector1 = GPTVectorStoreIndex.from_documents(documents[0:2], **vector_kwargs)
    vector2 = GPTVectorStoreIndex.from_documents(documents[2:4], **vector_kwargs)
    list3 = GPTVectorStoreIndex.from_documents(documents[4:6], **vector_kwargs)
    list4 = GPTVectorStoreIndex.from_documents(documents[6:8], **vector_kwargs)

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
) -> None:
    """Test query."""
    vector_kwargs = index_kwargs["vector"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    # try building a list for every two, then a tree
    vector1 = GPTVectorStoreIndex.from_documents(documents[0:2], **vector_kwargs)
    vector2 = GPTVectorStoreIndex.from_documents(documents[2:4], **vector_kwargs)
    list3 = GPTVectorStoreIndex.from_documents(documents[4:6], **vector_kwargs)
    list4 = GPTVectorStoreIndex.from_documents(documents[6:8], **vector_kwargs)

    summary1 = "foo bar"
    summary2 = "apple orange"
    summary3 = "toronto london"
    summary4 = "cat dog"
    summaries = [summary1, summary2, summary3, summary4]

    graph = ComposableGraph.from_indices(
        GPTVectorStoreIndex,
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


def get_pinecone_storage_context() -> StorageContext:
    # NOTE: mock pinecone import
    sys.modules["pinecone"] = MagicMock()
    return StorageContext.from_defaults(
        vector_store=PineconeVectorStore(pinecone_index=MockPineconeIndex())
    )


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
) -> None:
    """Test composing pinecone index on top of pinecone index."""
    pinecone_kwargs = index_kwargs["pinecone"]
    # try building a tree for a group of 4, then a list
    # use a diff set of documents
    # try building a list for every two, then a tree
    pinecone1 = GPTVectorStoreIndex.from_documents(
        documents[0:2],
        storage_context=get_pinecone_storage_context(),
        tokenizer=mock_tokenizer,
        **pinecone_kwargs
    )
    pinecone2 = GPTVectorStoreIndex.from_documents(
        documents[2:4],
        storage_context=get_pinecone_storage_context(),
        tokenizer=mock_tokenizer,
        **pinecone_kwargs
    )
    pinecone3 = GPTVectorStoreIndex.from_documents(
        documents[4:6],
        storage_context=get_pinecone_storage_context(),
        tokenizer=mock_tokenizer,
        **pinecone_kwargs
    )
    pinecone4 = GPTVectorStoreIndex.from_documents(
        documents[6:8],
        storage_context=get_pinecone_storage_context(),
        tokenizer=mock_tokenizer,
        **pinecone_kwargs
    )

    summary1 = "foo bar"
    summary2 = "apple orange"
    summary3 = "toronto london"
    summary4 = "cat dog"
    summaries = [summary1, summary2, summary3, summary4]

    graph = ComposableGraph.from_indices(
        GPTVectorStoreIndex,
        [pinecone1, pinecone2, pinecone3, pinecone4],
        index_summaries=summaries,
        storage_context=get_pinecone_storage_context(),
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
