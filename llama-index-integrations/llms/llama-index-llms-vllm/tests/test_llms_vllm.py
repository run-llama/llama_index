from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle


def test_embedding_class():
    from llama_index.llms.vllm import Vllm

    names_of_base_classes = [b.__name__ for b in Vllm.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_server_class():
    from llama_index.llms.vllm import VllmServer

    names_of_base_classes = [b.__name__ for b in VllmServer.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_server_callback() -> None:
    from llama_index.llms.vllm import VllmServer

    callback_manager = CallbackManager()
    remote = VllmServer(
        api_url="http://localhost:8000",
        model="modelstub",
        max_new_tokens=123,
        callback_manager=callback_manager,
    )
    assert remote.callback_manager == callback_manager
    del remote


# ---------------------------------------------------------------------------
# VllmLLM tests
# ---------------------------------------------------------------------------


def test_vllm_llm_class_hierarchy():
    """VllmLLM must be a proper LLM subclass."""
    from llama_index.llms.vllm import VllmLLM

    names_of_base_classes = [b.__name__ for b in VllmLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_vllm_llm_default_fields():
    """VllmLLM instantiates with sensible defaults (no real server needed)."""
    from llama_index.llms.vllm import VllmLLM

    with patch("openai.OpenAI"), patch("openai.AsyncOpenAI"):
        llm = VllmLLM(model="test-model", api_base="http://localhost:8000/v1")

    assert llm.model == "test-model"
    assert llm.api_base == "http://localhost:8000/v1"
    assert llm.is_chat_model is True
    assert llm.temperature == 0.1
    assert llm.max_tokens == 512


def test_vllm_llm_class_name():
    from llama_index.llms.vllm import VllmLLM

    assert VllmLLM.class_name() == "VllmLLM"


def test_vllm_llm_metadata():
    from llama_index.llms.vllm import VllmLLM

    with patch("openai.OpenAI"), patch("openai.AsyncOpenAI"):
        llm = VllmLLM(model="my-model", context_window=8192)

    meta = llm.metadata
    assert meta.model_name == "my-model"
    assert meta.is_chat_model is True
    assert meta.context_window == 8192


def test_vllm_llm_complete(monkeypatch):
    """complete() should call the OpenAI chat endpoint and return text."""
    from llama_index.llms.vllm import VllmLLM

    # Build a mock completion response
    mock_choice = MagicMock()
    mock_choice.message.content = "Paris"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model_dump.return_value = {}

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_openai), patch("openai.AsyncOpenAI"):
        llm = VllmLLM(model="test-model")
        result = llm.complete("What is the capital of France?")

    assert "Paris" in result.text


def test_vllm_llm_additional_kwargs():
    """additional_kwargs are stored and forwarded."""
    from llama_index.llms.vllm import VllmLLM

    with patch("openai.OpenAI"), patch("openai.AsyncOpenAI"):
        llm = VllmLLM(
            model="test",
            additional_kwargs={"guided_json": {"type": "string"}},
        )
    assert "guided_json" in llm.additional_kwargs


# ---------------------------------------------------------------------------
# VllmEmbedding tests
# ---------------------------------------------------------------------------


def test_vllm_embedding_class_hierarchy():
    """VllmEmbedding must be a proper BaseEmbedding subclass."""
    from llama_index.llms.vllm import VllmEmbedding

    names_of_base_classes = [b.__name__ for b in VllmEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


def test_vllm_embedding_class_name():
    from llama_index.llms.vllm import VllmEmbedding

    assert VllmEmbedding.class_name() == "VllmEmbedding"


def test_vllm_embedding_default_fields():
    from llama_index.llms.vllm import VllmEmbedding

    with patch("openai.OpenAI"), patch("openai.AsyncOpenAI"):
        emb = VllmEmbedding(
            model_name="BAAI/bge-base-en-v1.5",
            api_base="http://localhost:8000/v1",
        )

    assert emb.model_name == "BAAI/bge-base-en-v1.5"
    assert emb.api_base == "http://localhost:8000/v1"
    assert emb.api_key == "EMPTY"
    assert emb.dimensions is None


def test_vllm_embedding_get_text_embedding(monkeypatch):
    """_get_text_embedding() should call the OpenAI embeddings endpoint."""
    from llama_index.llms.vllm import VllmEmbedding

    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1, 0.2, 0.3]
    mock_response = MagicMock()
    mock_response.data = [mock_embedding_data]

    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_openai), patch("openai.AsyncOpenAI"):
        emb = VllmEmbedding(model_name="test-embed")
        result = emb.get_text_embedding("hello world")

    assert result == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# VllmRerank tests
# ---------------------------------------------------------------------------


def test_vllm_rerank_class_hierarchy():
    """VllmRerank must be a proper BaseNodePostprocessor subclass."""
    from llama_index.llms.vllm import VllmRerank

    names_of_base_classes = [b.__name__ for b in VllmRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


def test_vllm_rerank_class_name():
    from llama_index.llms.vllm import VllmRerank

    assert VllmRerank.class_name() == "VllmRerank"


def test_vllm_rerank_default_fields():
    from llama_index.llms.vllm import VllmRerank

    reranker = VllmRerank(model="BAAI/bge-reranker-base")
    assert reranker.model == "BAAI/bge-reranker-base"
    assert reranker.api_base == "http://localhost:8000"
    assert reranker.top_n == 3


def test_vllm_rerank_empty_nodes():
    """Should return an empty list when no nodes are provided."""
    from llama_index.llms.vllm import VllmRerank

    reranker = VllmRerank(model="test-model")
    result = reranker.postprocess_nodes(
        [], query_bundle=QueryBundle(query_str="test query")
    )
    assert result == []


def test_vllm_rerank_postprocess_nodes(monkeypatch):
    """postprocess_nodes() calls the rerank API and returns sorted results."""
    from llama_index.llms.vllm import VllmRerank

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.4},
        ]
    }

    nodes = [
        NodeWithScore(node=TextNode(text="doc A"), score=0.5),
        NodeWithScore(node=TextNode(text="doc B"), score=0.5),
    ]
    query = QueryBundle(query_str="test query")

    with patch("requests.post", return_value=mock_response):
        reranker = VllmRerank(model="reranker-model", top_n=2)
        result = reranker.postprocess_nodes(nodes, query_bundle=query)

    assert len(result) == 2
    assert result[0].score == pytest.approx(0.9)
    assert result[0].node.text == "doc B"
    assert result[1].score == pytest.approx(0.4)
    assert result[1].node.text == "doc A"


def test_vllm_rerank_requires_query():
    """Should raise ValueError when no query_bundle is provided."""
    from llama_index.llms.vllm import VllmRerank

    reranker = VllmRerank(model="test-model")
    with pytest.raises(ValueError, match="requires a query bundle"):
        reranker.postprocess_nodes(
            [NodeWithScore(node=TextNode(text="doc"), score=0.5)]
        )
