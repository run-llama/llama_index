"""Comprehensive functional tests for the LangChain embeddings adapter."""

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap mock modules so we never need real langchain installed.
# ---------------------------------------------------------------------------

# First, remove any pre-existing langchain modules that may have been loaded
# from the real (incompatible) langchain package so our stubs take effect.
for _key in list(sys.modules):
    if _key.startswith(("langchain", "langchain_core", "langchain_community")):
        del sys.modules[_key]

# Also remove any cached bridge/adapter modules so they re-import from our stubs.
for _key in list(sys.modules):
    if "llama_index.core.bridge.langchain" in _key:
        del sys.modules[_key]
    if "llama_index.embeddings.langchain" in _key:
        del sys.modules[_key]

# --- langchain_core stubs ---
_langchain_core = ModuleType("langchain_core")
_langchain_core.__path__ = []  # Mark as package
_langchain_core_messages = ModuleType("langchain_core.messages")


class _BaseMessage:
    def __init__(self, content="", additional_kwargs=None, **kw):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}

    @classmethod
    def schema(cls):
        return {"required": ["content"]}


class _HumanMessage(_BaseMessage):
    pass


class _AIMessage(_BaseMessage):
    pass


class _SystemMessage(_BaseMessage):
    pass


class _FunctionMessage(_BaseMessage):
    @classmethod
    def schema(cls):
        return {"required": ["content", "name"]}


class _ChatMessage(_BaseMessage):
    @classmethod
    def schema(cls):
        return {"required": ["content", "role"]}


_langchain_core_messages.BaseMessage = _BaseMessage
_langchain_core_messages.HumanMessage = _HumanMessage
_langchain_core_messages.AIMessage = _AIMessage
_langchain_core_messages.SystemMessage = _SystemMessage
_langchain_core_messages.FunctionMessage = _FunctionMessage
_langchain_core_messages.ChatMessage = _ChatMessage

sys.modules["langchain_core"] = _langchain_core
sys.modules["langchain_core.messages"] = _langchain_core_messages

# --- langchain stubs ---
_langchain = ModuleType("langchain")
_langchain.__path__ = []  # Mark as package
_langchain_base_language = ModuleType("langchain.base_language")
_langchain_schema = ModuleType("langchain.schema")
_langchain_schema.__path__ = []  # Mark as package (has sub-modules)
_langchain_chat_models = ModuleType("langchain.chat_models")
_langchain_chat_models.__path__ = []  # Mark as package
_langchain_chat_models_base = ModuleType("langchain.chat_models.base")


class _BaseLanguageModel:
    pass


class _BaseChatModel(_BaseLanguageModel):
    pass


_langchain_base_language.BaseLanguageModel = _BaseLanguageModel
_langchain_schema.AIMessage = _AIMessage
_langchain_schema.BaseMessage = _BaseMessage
_langchain_schema.HumanMessage = _HumanMessage
_langchain_schema.SystemMessage = _SystemMessage
_langchain_schema.FunctionMessage = _FunctionMessage
_langchain_schema.ChatMessage = _ChatMessage
_langchain_schema.BaseMemory = type("BaseMemory", (), {})
_langchain_schema.BaseOutputParser = type("BaseOutputParser", (), {})
_langchain_schema.ChatGeneration = type("ChatGeneration", (), {})
_langchain_schema.LLMResult = type("LLMResult", (), {})
_langchain_chat_models_base.BaseChatModel = _BaseChatModel

sys.modules["langchain"] = _langchain
sys.modules["langchain.base_language"] = _langchain_base_language
sys.modules["langchain.schema"] = _langchain_schema
sys.modules["langchain.chat_models"] = _langchain_chat_models
sys.modules["langchain.chat_models.base"] = _langchain_chat_models_base

# --- langchain_community stubs ---
_langchain_community = ModuleType("langchain_community")
_langchain_community.__path__ = []  # Mark as package
_langchain_community_chat_models = ModuleType("langchain_community.chat_models")
_langchain_community_llms = ModuleType("langchain_community.llms")
_langchain_community_embeddings = ModuleType("langchain_community.embeddings")
_langchain_community_chat_message_histories = ModuleType(
    "langchain_community.chat_message_histories"
)


class _ChatOpenAI(_BaseChatModel):
    pass


class _ChatAnyscale(_BaseChatModel):
    pass


class _ChatFireworks(_BaseChatModel):
    pass


class _OpenAI(_BaseLanguageModel):
    pass


class _Cohere(_BaseLanguageModel):
    pass


class _AI21(_BaseLanguageModel):
    pass


class _BaseLLM(_BaseLanguageModel):
    pass


class _FakeListLLM(_BaseLLM):
    pass


_langchain_community_chat_models.ChatOpenAI = _ChatOpenAI
_langchain_community_chat_models.ChatAnyscale = _ChatAnyscale
_langchain_community_chat_models.ChatFireworks = _ChatFireworks
_langchain_community_llms.OpenAI = _OpenAI
_langchain_community_llms.AI21 = _AI21
_langchain_community_llms.Cohere = _Cohere
_langchain_community_llms.BaseLLM = _BaseLLM
_langchain_community_llms.FakeListLLM = _FakeListLLM
_langchain_community_embeddings.HuggingFaceEmbeddings = type(
    "HuggingFaceEmbeddings", (), {}
)
_langchain_community_embeddings.HuggingFaceBgeEmbeddings = type(
    "HuggingFaceBgeEmbeddings", (), {}
)
_langchain_community_chat_message_histories.ChatMessageHistory = type(
    "ChatMessageHistory", (), {}
)

sys.modules["langchain_community"] = _langchain_community
sys.modules["langchain_community.chat_models"] = _langchain_community_chat_models
sys.modules["langchain_community.llms"] = _langchain_community_llms
sys.modules["langchain_community.embeddings"] = _langchain_community_embeddings
sys.modules["langchain_community.chat_message_histories"] = (
    _langchain_community_chat_message_histories
)

# Remaining langchain sub-modules referenced by the bridge
for _mod_path in [
    "langchain.agents",
    "langchain.agents.agent_toolkits",
    "langchain.agents.agent_toolkits.base",
    "langchain.callbacks",
    "langchain.callbacks.base",
    "langchain.chains",
    "langchain.chains.prompt_selector",
    "langchain.docstore",
    "langchain.docstore.document",
    "langchain.memory",
    "langchain.memory.chat_memory",
    "langchain.output_parsers",
    "langchain.prompts",
    "langchain.prompts.chat",
    "langchain.schema.embeddings",
    "langchain.schema.prompt_template",
    "langchain.text_splitter",
    "langchain.tools",
]:
    mod = ModuleType(_mod_path)
    mod.__path__ = []  # Mark all as packages in case they have sub-modules
    mod.__dict__.setdefault("BaseCallbackHandler", type("BaseCallbackHandler", (), {}))
    mod.__dict__.setdefault("BaseCallbackManager", type("BaseCallbackManager", (), {}))
    mod.__dict__.setdefault("LLMResult", type("LLMResult", (), {}))
    mod.__dict__.setdefault("AgentExecutor", type("AgentExecutor", (), {}))
    mod.__dict__.setdefault("AgentType", type("AgentType", (), {}))
    mod.__dict__.setdefault("initialize_agent", lambda *a, **kw: None)
    mod.__dict__.setdefault("BaseToolkit", type("BaseToolkit", (), {}))
    mod.__dict__.setdefault(
        "ConditionalPromptSelector", type("ConditionalPromptSelector", (), {})
    )
    mod.__dict__.setdefault("is_chat_model", lambda m: False)
    mod.__dict__.setdefault("Document", type("Document", (), {}))
    mod.__dict__.setdefault(
        "ConversationBufferMemory", type("ConversationBufferMemory", (), {})
    )
    mod.__dict__.setdefault("BaseChatMemory", type("BaseChatMemory", (), {}))
    mod.__dict__.setdefault("ResponseSchema", type("ResponseSchema", (), {}))
    mod.__dict__.setdefault("PromptTemplate", type("PromptTemplate", (), {}))
    mod.__dict__.setdefault("BasePromptTemplate", type("BasePromptTemplate", (), {}))
    mod.__dict__.setdefault(
        "AIMessagePromptTemplate", type("AIMessagePromptTemplate", (), {})
    )
    mod.__dict__.setdefault("ChatPromptTemplate", type("ChatPromptTemplate", (), {}))
    mod.__dict__.setdefault(
        "HumanMessagePromptTemplate", type("HumanMessagePromptTemplate", (), {})
    )
    mod.__dict__.setdefault(
        "BaseMessagePromptTemplate", type("BaseMessagePromptTemplate", (), {})
    )
    mod.__dict__.setdefault(
        "SystemMessagePromptTemplate", type("SystemMessagePromptTemplate", (), {})
    )
    mod.__dict__.setdefault("BaseMemory", type("BaseMemory", (), {}))
    mod.__dict__.setdefault("BaseOutputParser", type("BaseOutputParser", (), {}))
    mod.__dict__.setdefault("ChatGeneration", type("ChatGeneration", (), {}))
    mod.__dict__.setdefault("Embeddings", type("Embeddings", (), {}))
    mod.__dict__.setdefault(
        "RecursiveCharacterTextSplitter",
        type("RecursiveCharacterTextSplitter", (), {}),
    )
    mod.__dict__.setdefault("TextSplitter", type("TextSplitter", (), {}))
    mod.__dict__.setdefault("BaseTool", type("BaseTool", (), {}))
    mod.__dict__.setdefault("StructuredTool", type("StructuredTool", (), {}))
    mod.__dict__.setdefault("Tool", type("Tool", (), {}))
    sys.modules[_mod_path] = mod

# ---------------------------------------------------------------------------
# Now import the actual adapter code under test.
# ---------------------------------------------------------------------------
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding


# ===================================================================
# Helpers
# ===================================================================


def _make_mock_embeddings(**attrs):
    """Create a mock LangChain Embeddings object."""
    mock = MagicMock()
    for k, v in attrs.items():
        setattr(mock, k, v)
    return mock


def _build_embedding_adapter(mock_embeddings, model_name=None):
    """Build a LangchainEmbedding adapter wrapping the given mock.

    Uses __new__ + direct attr setting to bypass pydantic __init__ validation
    that may try to access real langchain.
    """
    adapter = LangchainEmbedding.__new__(LangchainEmbedding)
    adapter._langchain_embedding = mock_embeddings
    adapter._async_not_implemented_warned = False

    # Determine model_name the same way the real __init__ does
    if model_name is not None:
        resolved_name = model_name
    elif hasattr(mock_embeddings, "model_name") and mock_embeddings.model_name:
        resolved_name = mock_embeddings.model_name
    elif hasattr(mock_embeddings, "model") and mock_embeddings.model:
        resolved_name = mock_embeddings.model
    else:
        resolved_name = type(mock_embeddings).__name__

    adapter.__dict__.update(
        {
            "model_name": resolved_name,
            "embed_batch_size": 10,
            "callback_manager": MagicMock(),
        }
    )
    return adapter


# ===================================================================
# 1. Inheritance test
# ===================================================================


def test_langchain_embedding_class():
    """LangchainEmbedding must be a subclass of BaseEmbedding."""
    names_of_base_classes = [b.__name__ for b in LangchainEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


# ===================================================================
# 2. test_get_query_embedding
# ===================================================================


def test_get_query_embedding():
    """_get_query_embedding should delegate to embed_query."""
    mock_emb = _make_mock_embeddings()
    mock_emb.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])

    adapter = _build_embedding_adapter(mock_emb)
    result = adapter._get_query_embedding("test query")

    assert result == [0.1, 0.2, 0.3]
    mock_emb.embed_query.assert_called_once_with("test query")


def test_get_query_embedding_returns_correct_dimensions():
    """_get_query_embedding should return the full embedding vector."""
    embedding = [float(i) for i in range(768)]
    mock_emb = _make_mock_embeddings()
    mock_emb.embed_query = MagicMock(return_value=embedding)

    adapter = _build_embedding_adapter(mock_emb)
    result = adapter._get_query_embedding("a sentence")

    assert len(result) == 768
    assert result == embedding


# ===================================================================
# 3. test_get_text_embedding
# ===================================================================


def test_get_text_embedding():
    """_get_text_embedding should call embed_documents with a single-element list."""
    mock_emb = _make_mock_embeddings()
    mock_emb.embed_documents = MagicMock(return_value=[[0.4, 0.5, 0.6]])

    adapter = _build_embedding_adapter(mock_emb)
    result = adapter._get_text_embedding("some text")

    assert result == [0.4, 0.5, 0.6]
    mock_emb.embed_documents.assert_called_once_with(["some text"])


def test_get_text_embedding_extracts_first_element():
    """_get_text_embedding should return only the first element from embed_documents."""
    mock_emb = _make_mock_embeddings()
    mock_emb.embed_documents = MagicMock(return_value=[[1.0, 2.0], [3.0, 4.0]])

    adapter = _build_embedding_adapter(mock_emb)
    result = adapter._get_text_embedding("text")

    # Should only return the first embedding
    assert result == [1.0, 2.0]


# ===================================================================
# 4. test_get_text_embeddings_batch
# ===================================================================


def test_get_text_embeddings_batch():
    """_get_text_embeddings should call embed_documents with the full list."""
    mock_emb = _make_mock_embeddings()
    expected = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    mock_emb.embed_documents = MagicMock(return_value=expected)

    adapter = _build_embedding_adapter(mock_emb)
    texts = ["text1", "text2", "text3"]
    result = adapter._get_text_embeddings(texts)

    assert result == expected
    mock_emb.embed_documents.assert_called_once_with(texts)


def test_get_text_embeddings_batch_empty():
    """_get_text_embeddings should handle an empty list correctly."""
    mock_emb = _make_mock_embeddings()
    mock_emb.embed_documents = MagicMock(return_value=[])

    adapter = _build_embedding_adapter(mock_emb)
    result = adapter._get_text_embeddings([])

    assert result == []
    mock_emb.embed_documents.assert_called_once_with([])


# ===================================================================
# 5. test_aget_query_embedding_native
# ===================================================================


@pytest.mark.asyncio
async def test_aget_query_embedding_native():
    """_aget_query_embedding should use aembed_query when it works."""
    mock_emb = _make_mock_embeddings()
    mock_emb.aembed_query = AsyncMock(return_value=[0.7, 0.8, 0.9])

    adapter = _build_embedding_adapter(mock_emb)
    result = await adapter._aget_query_embedding("async query")

    assert result == [0.7, 0.8, 0.9]
    mock_emb.aembed_query.assert_awaited_once_with("async query")


# ===================================================================
# 6. test_aget_query_embedding_fallback
# ===================================================================


@pytest.mark.asyncio
async def test_aget_query_embedding_fallback():
    """_aget_query_embedding should fall back to sync when aembed_query raises NotImplementedError."""
    mock_emb = _make_mock_embeddings()
    mock_emb.aembed_query = AsyncMock(side_effect=NotImplementedError)
    mock_emb.embed_query = MagicMock(return_value=[1.0, 1.1, 1.2])

    adapter = _build_embedding_adapter(mock_emb)
    result = await adapter._aget_query_embedding("fallback query")

    assert result == [1.0, 1.1, 1.2]
    mock_emb.embed_query.assert_called_once_with("fallback query")


@pytest.mark.asyncio
async def test_aget_query_embedding_fallback_warns_once():
    """The fallback should print a warning, but only once."""
    mock_emb = _make_mock_embeddings()
    mock_emb.aembed_query = AsyncMock(side_effect=NotImplementedError)
    mock_emb.embed_query = MagicMock(return_value=[0.0])

    adapter = _build_embedding_adapter(mock_emb)
    assert adapter._async_not_implemented_warned is False

    with patch("builtins.print") as mock_print:
        await adapter._aget_query_embedding("q1")
        assert adapter._async_not_implemented_warned is True
        mock_print.assert_called_once()

        # Second call should NOT print again
        mock_print.reset_mock()
        await adapter._aget_query_embedding("q2")
        mock_print.assert_not_called()


# ===================================================================
# 7. test_aget_text_embedding_native
# ===================================================================


@pytest.mark.asyncio
async def test_aget_text_embedding_native():
    """_aget_text_embedding should use aembed_documents when it works."""
    mock_emb = _make_mock_embeddings()
    mock_emb.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    adapter = _build_embedding_adapter(mock_emb)
    result = await adapter._aget_text_embedding("async text")

    assert result == [0.1, 0.2, 0.3]
    mock_emb.aembed_documents.assert_awaited_once_with(["async text"])


# ===================================================================
# 8. test_aget_text_embedding_fallback
# ===================================================================


@pytest.mark.asyncio
async def test_aget_text_embedding_fallback():
    """_aget_text_embedding should fall back to sync when aembed_documents raises NotImplementedError."""
    mock_emb = _make_mock_embeddings()
    mock_emb.aembed_documents = AsyncMock(side_effect=NotImplementedError)
    mock_emb.embed_documents = MagicMock(return_value=[[0.5, 0.6]])

    adapter = _build_embedding_adapter(mock_emb)
    result = await adapter._aget_text_embedding("fallback text")

    assert result == [0.5, 0.6]
    mock_emb.embed_documents.assert_called_once_with(["fallback text"])


@pytest.mark.asyncio
async def test_aget_text_embedding_fallback_warns():
    """The fallback should trigger the warn-once mechanism."""
    mock_emb = _make_mock_embeddings()
    mock_emb.aembed_documents = AsyncMock(side_effect=NotImplementedError)
    mock_emb.embed_documents = MagicMock(return_value=[[0.0]])

    adapter = _build_embedding_adapter(mock_emb)
    assert adapter._async_not_implemented_warned is False

    with patch("builtins.print"):
        await adapter._aget_text_embedding("text")
        assert adapter._async_not_implemented_warned is True


# ===================================================================
# 9. test_aget_text_embeddings_batch_native
# ===================================================================


@pytest.mark.asyncio
async def test_aget_text_embeddings_batch_native():
    """_aget_text_embeddings should use aembed_documents when it works."""
    mock_emb = _make_mock_embeddings()
    expected = [[0.1, 0.2], [0.3, 0.4]]
    mock_emb.aembed_documents = AsyncMock(return_value=expected)

    adapter = _build_embedding_adapter(mock_emb)
    result = await adapter._aget_text_embeddings(["text1", "text2"])

    assert result == expected
    mock_emb.aembed_documents.assert_awaited_once_with(["text1", "text2"])


# ===================================================================
# 10. test_aget_text_embeddings_batch_fallback
# ===================================================================


@pytest.mark.asyncio
async def test_aget_text_embeddings_batch_fallback():
    """_aget_text_embeddings should fall back to sync when aembed_documents raises NotImplementedError."""
    mock_emb = _make_mock_embeddings()
    expected = [[0.5, 0.6], [0.7, 0.8]]
    mock_emb.aembed_documents = AsyncMock(side_effect=NotImplementedError)
    mock_emb.embed_documents = MagicMock(return_value=expected)

    adapter = _build_embedding_adapter(mock_emb)
    result = await adapter._aget_text_embeddings(["t1", "t2"])

    assert result == expected
    mock_emb.embed_documents.assert_called_once_with(["t1", "t2"])


@pytest.mark.asyncio
async def test_aget_text_embeddings_batch_fallback_warns():
    """The batch fallback should trigger the warn-once mechanism."""
    mock_emb = _make_mock_embeddings()
    mock_emb.aembed_documents = AsyncMock(side_effect=NotImplementedError)
    mock_emb.embed_documents = MagicMock(return_value=[[0.0]])

    adapter = _build_embedding_adapter(mock_emb)
    assert adapter._async_not_implemented_warned is False

    with patch("builtins.print"):
        await adapter._aget_text_embeddings(["text"])
        assert adapter._async_not_implemented_warned is True


# ===================================================================
# 11. test_model_name_auto_detection
# ===================================================================


def test_model_name_auto_detection_model_name():
    """Model name should be extracted from model_name attribute."""
    mock_emb = _make_mock_embeddings(model_name="text-embedding-ada-002")
    adapter = _build_embedding_adapter(mock_emb)
    assert adapter.model_name == "text-embedding-ada-002"


def test_model_name_auto_detection_model():
    """Model name should fall back to model attribute."""
    mock_emb = MagicMock(spec=[])
    mock_emb.model = "all-MiniLM-L6-v2"
    adapter = _build_embedding_adapter(mock_emb)
    assert adapter.model_name == "all-MiniLM-L6-v2"


def test_model_name_auto_detection_class_name():
    """Model name should fall back to class name when no model attrs exist."""

    class CustomEmbeddings:
        pass

    mock_emb = CustomEmbeddings()
    adapter = _build_embedding_adapter(mock_emb)
    assert adapter.model_name == "CustomEmbeddings"


def test_model_name_explicit_override():
    """Explicitly passed model_name should override auto-detection."""
    mock_emb = _make_mock_embeddings(model_name="auto-detected")
    adapter = _build_embedding_adapter(mock_emb, model_name="explicit-name")
    assert adapter.model_name == "explicit-name"


# ===================================================================
# Additional edge-case tests
# ===================================================================


def test_class_name():
    """LangchainEmbedding.class_name() should return the expected string."""
    assert LangchainEmbedding.class_name() == "LangchainEmbedding"


def test_warn_once_flag_is_initially_false():
    """The async warning flag should start as False."""
    mock_emb = _make_mock_embeddings()
    adapter = _build_embedding_adapter(mock_emb)
    assert adapter._async_not_implemented_warned is False


@pytest.mark.asyncio
async def test_warn_once_flag_shared_across_methods():
    """Once one async method falls back, subsequent fallbacks should not warn again."""
    mock_emb = _make_mock_embeddings()
    mock_emb.aembed_query = AsyncMock(side_effect=NotImplementedError)
    mock_emb.aembed_documents = AsyncMock(side_effect=NotImplementedError)
    mock_emb.embed_query = MagicMock(return_value=[0.0])
    mock_emb.embed_documents = MagicMock(return_value=[[0.0]])

    adapter = _build_embedding_adapter(mock_emb)

    with patch("builtins.print") as mock_print:
        # First fallback triggers print
        await adapter._aget_query_embedding("q")
        assert mock_print.call_count == 1

        # Second fallback (different method) should NOT print
        await adapter._aget_text_embedding("t")
        assert mock_print.call_count == 1

        # Third fallback (batch) should also NOT print
        await adapter._aget_text_embeddings(["t1"])
        assert mock_print.call_count == 1


def test_get_text_embeddings_preserves_order():
    """_get_text_embeddings should preserve the order of input texts."""
    mock_emb = _make_mock_embeddings()
    embeddings = [[float(i)] for i in range(5)]
    mock_emb.embed_documents = MagicMock(return_value=embeddings)

    adapter = _build_embedding_adapter(mock_emb)
    texts = [f"text_{i}" for i in range(5)]
    result = adapter._get_text_embeddings(texts)

    assert result == embeddings
    mock_emb.embed_documents.assert_called_once_with(texts)


@pytest.mark.asyncio
async def test_aget_text_embedding_extracts_first_element():
    """_aget_text_embedding should return only the first embedding from aembed_documents."""
    mock_emb = _make_mock_embeddings()
    mock_emb.aembed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

    adapter = _build_embedding_adapter(mock_emb)
    result = await adapter._aget_text_embedding("text")

    # Should return only the first element
    assert result == [0.1, 0.2]
