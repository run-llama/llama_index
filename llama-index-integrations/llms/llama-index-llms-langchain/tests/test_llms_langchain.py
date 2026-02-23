"""Comprehensive functional tests for the LangChain LLM adapter."""

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap mock modules so we never need real langchain installed.
# We create lightweight stubs for the langchain classes that the adapter and
# its utility module reference at import time.
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
    if "llama_index.llms.langchain" in _key:
        del sys.modules[_key]

# --- langchain_core stubs ---
_langchain_core = ModuleType("langchain_core")
_langchain_core.__path__ = []  # Mark as package
_langchain_core_messages = ModuleType("langchain_core.messages")


class _BaseMessage:
    """Minimal LangChain BaseMessage mock."""

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
    def __init__(self, content="", name="", additional_kwargs=None, **kw):
        super().__init__(content=content, additional_kwargs=additional_kwargs)
        self.name = name

    @classmethod
    def schema(cls):
        return {"required": ["content", "name"]}


class _ChatMessage(_BaseMessage):
    def __init__(self, content="", role="", additional_kwargs=None, **kw):
        super().__init__(content=content, additional_kwargs=additional_kwargs)
        self.role = role

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
    """Stand-in for langchain.base_language.BaseLanguageModel."""
    pass


class _BaseChatModel(_BaseLanguageModel):
    """Stand-in for langchain.chat_models.base.BaseChatModel."""
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

# --- langchain_community stubs (needed by bridge) ---
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
    # Provide dummy attrs so the bridge import doesn't crash
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
    mod.__dict__.setdefault("ConversationBufferMemory", type("ConversationBufferMemory", (), {}))
    mod.__dict__.setdefault("BaseChatMemory", type("BaseChatMemory", (), {}))
    mod.__dict__.setdefault("ResponseSchema", type("ResponseSchema", (), {}))
    mod.__dict__.setdefault("PromptTemplate", type("PromptTemplate", (), {}))
    mod.__dict__.setdefault("BasePromptTemplate", type("BasePromptTemplate", (), {}))
    mod.__dict__.setdefault("AIMessagePromptTemplate", type("AIMessagePromptTemplate", (), {}))
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
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.langchain.utils import (
    _extract_context_window,
    _extract_max_tokens,
    _extract_model_name,
    from_lc_messages,
    get_llm_metadata,
    is_chat_model,
    to_lc_messages,
)


# ===================================================================
# Helpers
# ===================================================================


def _make_chat_model(**extra_attrs):
    """Create a mock that is an instance of BaseChatModel (chat model)."""
    mock = MagicMock(spec=_BaseChatModel)
    mock.__class__ = type(
        "MockChatModel", (_BaseChatModel,), {}
    )
    for k, v in extra_attrs.items():
        setattr(mock, k, v)
    return mock


def _make_plain_llm(**extra_attrs):
    """Create a mock that is a BaseLanguageModel but NOT a BaseChatModel."""
    mock = MagicMock(spec=_BaseLanguageModel)
    mock.__class__ = type(
        "MockPlainLLM", (_BaseLanguageModel,), {}
    )
    for k, v in extra_attrs.items():
        setattr(mock, k, v)
    return mock


def _build_adapter(mock_llm):
    """Build a LangChainLLM adapter wrapping the given mock.

    Uses model_construct to properly initialize pydantic v2 internals,
    then sets the private _llm attribute.
    """
    from llama_index.core.callbacks import CallbackManager

    adapter = LangChainLLM.model_construct(
        system_prompt=None,
        messages_to_prompt=None,
        completion_to_prompt=None,
        output_parser=None,
        pydantic_program_mode="default",
        callback_manager=CallbackManager(),
    )
    # Set private attrs (not handled by model_construct)
    adapter._llm = mock_llm
    return adapter


# ===================================================================
# 1. Inheritance / class hierarchy
# ===================================================================


def test_langchain_llm_inherits_from_base_llm():
    """LangChainLLM must be a subclass of BaseLLM."""
    names_of_base_classes = [b.__name__ for b in LangChainLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


# ===================================================================
# 2. LangChainLLM metadata extraction with mock LLM
# ===================================================================


def test_metadata_property_delegates_to_get_llm_metadata():
    """The metadata property should delegate to get_llm_metadata."""
    mock_llm = _make_plain_llm()
    mock_llm.model_name = "test-model"
    mock_llm.max_tokens = 100
    mock_llm.n_ctx = 2048

    adapter = _build_adapter(mock_llm)

    with patch("llama_index.llms.langchain.utils.get_llm_metadata") as mock_get:
        mock_get.return_value = MagicMock(is_chat_model=False)
        _ = adapter.metadata
        mock_get.assert_called_once_with(mock_llm)


# ===================================================================
# 3. chat() for chat models (mock BaseChatModel with invoke())
# ===================================================================


def test_chat_chat_model_uses_invoke():
    """chat() on a chat model should call invoke with LC messages and return ChatResponse."""
    mock_llm = _make_chat_model()
    returned_ai = _AIMessage(content="Hello from LangChain!")
    mock_llm.invoke = MagicMock(return_value=returned_ai)

    adapter = _build_adapter(mock_llm)

    messages = [
        ChatMessage(role=MessageRole.USER, content="Hi"),
    ]

    response = adapter.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.content == "Hello from LangChain!"
    assert response.message.role == MessageRole.ASSISTANT
    mock_llm.invoke.assert_called_once()


# ===================================================================
# 4. chat() for non-chat models (falls back to complete)
# ===================================================================


def test_chat_non_chat_model_delegates_to_complete():
    """chat() on a non-chat model should delegate to complete() via messages_to_prompt."""
    mock_llm = _make_plain_llm()
    mock_llm.invoke = MagicMock(return_value="completion text")

    adapter = _build_adapter(mock_llm)
    adapter.messages_to_prompt = lambda msgs: "formatted prompt"

    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]
    response = adapter.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT


# ===================================================================
# 5. complete() with mock LLM
# ===================================================================


def test_complete_returns_completion_response():
    """complete() should invoke the LLM with a prompt string and return CompletionResponse."""
    mock_llm = _make_plain_llm()
    mock_llm.invoke = MagicMock(return_value="completed text")

    adapter = _build_adapter(mock_llm)

    response = adapter.complete("Say hello", formatted=True)

    assert isinstance(response, CompletionResponse)
    assert response.text == "completed text"
    mock_llm.invoke.assert_called_once()
    assert mock_llm.invoke.call_args[0][0] == "Say hello"


# ===================================================================
# 6. complete() with AIMessage return type handling
# ===================================================================


def test_complete_unwraps_ai_message():
    """complete() should unwrap AIMessage if the LLM returns one."""
    mock_llm = _make_plain_llm()
    mock_llm.invoke = MagicMock(return_value=_AIMessage(content="unwrapped"))

    adapter = _build_adapter(mock_llm)
    response = adapter.complete("prompt", formatted=True)

    assert response.text == "unwrapped"


# ===================================================================
# 7. stream_chat() with native stream() method
# ===================================================================


def test_stream_chat_native_stream():
    """stream_chat() should use .stream() when available on a chat model."""
    mock_llm = _make_chat_model()

    chunks = [
        _AIMessage(content="Hello"),
        _AIMessage(content=" world"),
    ]

    def _stream(messages, **kwargs):
        for c in chunks:
            yield c

    mock_llm.stream = _stream

    adapter = _build_adapter(mock_llm)
    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]

    gen = adapter.stream_chat(messages)
    results = list(gen)

    assert len(results) == 2
    assert results[0].delta == "Hello"
    assert results[1].delta == " world"
    assert results[1].message.content == "Hello world"


# ===================================================================
# 8. stream_chat() fallback with StreamingGeneratorCallbackHandler
# ===================================================================


def test_stream_chat_callback_handler_fallback():
    """
    stream_chat() should fall back to the callback-based gen_cb path when
    .stream() is absent but .streaming and .callbacks exist.
    """
    mock_llm = _make_chat_model()
    if hasattr(mock_llm, "stream"):
        del mock_llm.stream

    mock_llm.streaming = False
    mock_llm.callbacks = []

    returned_ai = _AIMessage(content="callback response")
    mock_llm.invoke = MagicMock(return_value=returned_ai)

    adapter = _build_adapter(mock_llm)
    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]

    with patch(
        "llama_index.core.langchain_helpers.streaming.StreamingGeneratorCallbackHandler"
    ) as MockHandler:
        handler_instance = MagicMock()
        handler_instance.get_response_gen.return_value = iter(["tok1", "tok2"])
        MockHandler.return_value = handler_instance

        gen = adapter.stream_chat(messages)
        results = list(gen)

    assert len(results) == 2
    assert results[0].delta == "tok1"
    assert results[1].delta == "tok2"
    assert results[1].message.content == "tok1tok2"
    assert results[0].message.role == MessageRole.ASSISTANT


def test_stream_chat_missing_streaming_raises():
    """stream_chat() callback path should raise if model lacks .streaming."""
    mock_llm = _make_chat_model()
    if hasattr(mock_llm, "stream"):
        del mock_llm.stream
    if hasattr(mock_llm, "streaming"):
        del mock_llm.streaming

    adapter = _build_adapter(mock_llm)
    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]

    with pytest.raises(ValueError, match="must support streaming"):
        list(adapter.stream_chat(messages))


def test_stream_chat_missing_callbacks_raises():
    """stream_chat() callback path should raise if model lacks .callbacks."""
    mock_llm = _make_chat_model()
    if hasattr(mock_llm, "stream"):
        del mock_llm.stream
    mock_llm.streaming = False
    if hasattr(mock_llm, "callbacks"):
        del mock_llm.callbacks

    adapter = _build_adapter(mock_llm)
    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]

    with pytest.raises(ValueError, match="must support callbacks"):
        list(adapter.stream_chat(messages))


# ===================================================================
# 9. stream_complete() with native stream() method
# ===================================================================


def test_stream_complete_native_stream():
    """stream_complete() should use .stream() when available."""
    mock_llm = _make_plain_llm()

    chunks = [_AIMessage(content="s1"), _AIMessage(content="s2"), _AIMessage(content="s3")]

    def _stream(prompt, **kwargs):
        for c in chunks:
            yield c

    mock_llm.stream = _stream

    adapter = _build_adapter(mock_llm)

    gen = adapter.stream_complete("prompt", formatted=True)
    results = list(gen)

    assert len(results) == 3
    assert results[0].delta == "s1"
    assert results[1].delta == "s2"
    assert results[2].delta == "s3"
    assert results[2].text == "s1s2s3"


def test_stream_complete_native_stream_plain_string():
    """stream_complete() should handle plain string chunks (no .content attr)."""
    mock_llm = _make_plain_llm()

    def _stream(prompt, **kwargs):
        for s in ["x", "y", "z"]:
            yield s

    mock_llm.stream = _stream

    adapter = _build_adapter(mock_llm)
    gen = adapter.stream_complete("prompt", formatted=True)
    results = list(gen)

    assert len(results) == 3
    assert results[0].delta == "x"
    assert results[2].text == "xyz"


# ===================================================================
# 10. stream_complete() fallback path
# ===================================================================


def test_stream_complete_callback_fallback():
    """
    stream_complete() should fall back to callback-based streaming when
    .stream() is absent.
    """
    mock_llm = _make_plain_llm()
    if hasattr(mock_llm, "stream"):
        del mock_llm.stream
    mock_llm.streaming = False
    mock_llm.callbacks = []
    mock_llm.invoke = MagicMock(return_value="callback complete text")

    adapter = _build_adapter(mock_llm)

    with patch(
        "llama_index.core.langchain_helpers.streaming.StreamingGeneratorCallbackHandler"
    ) as MockHandler:
        handler_instance = MagicMock()
        handler_instance.get_response_gen.return_value = iter(["a", "b", "c"])
        MockHandler.return_value = handler_instance

        gen = adapter.stream_complete("prompt", formatted=True)
        results = list(gen)

    assert len(results) == 3
    assert results[0].delta == "a"
    assert results[2].text == "abc"


def test_stream_complete_missing_streaming_raises():
    """stream_complete() callback path should raise when .streaming is absent."""
    mock_llm = _make_plain_llm()
    if hasattr(mock_llm, "stream"):
        del mock_llm.stream
    if hasattr(mock_llm, "streaming"):
        del mock_llm.streaming

    adapter = _build_adapter(mock_llm)

    with pytest.raises(ValueError, match="must support streaming"):
        list(adapter.stream_complete("prompt", formatted=True))


def test_stream_complete_missing_callbacks_raises():
    """stream_complete() callback path should raise when .callbacks is absent."""
    mock_llm = _make_plain_llm()
    if hasattr(mock_llm, "stream"):
        del mock_llm.stream
    mock_llm.streaming = False
    if hasattr(mock_llm, "callbacks"):
        del mock_llm.callbacks

    adapter = _build_adapter(mock_llm)

    with pytest.raises(ValueError, match="must support callbacks"):
        list(adapter.stream_complete("prompt", formatted=True))


# ===================================================================
# 11. achat() with native ainvoke() on chat model
# ===================================================================


@pytest.mark.asyncio
async def test_achat_native_ainvoke():
    """achat() should use ainvoke when available on a chat model."""
    mock_llm = _make_chat_model()
    returned_ai = _AIMessage(content="async hello")
    mock_llm.ainvoke = AsyncMock(return_value=returned_ai)

    adapter = _build_adapter(mock_llm)

    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]
    response = await adapter.achat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.content == "async hello"
    mock_llm.ainvoke.assert_awaited_once()


# ===================================================================
# 12. achat() fallback to thread pool
# ===================================================================


@pytest.mark.asyncio
async def test_achat_fallback_to_thread():
    """achat() should fall back to sync chat() via thread when ainvoke is absent."""
    mock_llm = _make_chat_model()
    if hasattr(mock_llm, "ainvoke"):
        del mock_llm.ainvoke

    returned_ai = _AIMessage(content="sync fallback")
    mock_llm.invoke = MagicMock(return_value=returned_ai)

    adapter = _build_adapter(mock_llm)
    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]

    response = await adapter.achat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.content == "sync fallback"


# ===================================================================
# 13. acomplete() with native ainvoke()
# ===================================================================


@pytest.mark.asyncio
async def test_acomplete_native_ainvoke():
    """acomplete() should use ainvoke when available."""
    mock_llm = _make_plain_llm()
    mock_llm.ainvoke = AsyncMock(return_value="async completed")

    adapter = _build_adapter(mock_llm)

    response = await adapter.acomplete("prompt", formatted=True)

    assert isinstance(response, CompletionResponse)
    assert response.text == "async completed"
    mock_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_acomplete_native_ainvoke_unwraps_ai_message():
    """acomplete() should unwrap AIMessage when ainvoke returns one."""
    mock_llm = _make_plain_llm()
    mock_llm.ainvoke = AsyncMock(return_value=_AIMessage(content="unwrapped async"))

    adapter = _build_adapter(mock_llm)
    response = await adapter.acomplete("prompt", formatted=True)

    assert response.text == "unwrapped async"


# ===================================================================
# 14. acomplete() fallback to thread pool
# ===================================================================


@pytest.mark.asyncio
async def test_acomplete_fallback_to_thread():
    """acomplete() should fall back to sync complete() via thread when ainvoke is missing."""
    mock_llm = _make_plain_llm()
    if hasattr(mock_llm, "ainvoke"):
        del mock_llm.ainvoke
    mock_llm.invoke = MagicMock(return_value="thread fallback")

    adapter = _build_adapter(mock_llm)

    response = await adapter.acomplete("prompt", formatted=True)

    assert isinstance(response, CompletionResponse)
    assert response.text == "thread fallback"


# ===================================================================
# 15. astream_chat() with native astream()
# ===================================================================


@pytest.mark.asyncio
async def test_astream_chat_native_astream():
    """astream_chat() should use native astream when available on a chat model."""
    mock_llm = _make_chat_model()

    chunks = [
        _AIMessage(content="Hello"),
        _AIMessage(content=" world"),
    ]

    async def _astream(messages, **kwargs):
        for c in chunks:
            yield c

    mock_llm.astream = _astream

    adapter = _build_adapter(mock_llm)
    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]

    gen = await adapter.astream_chat(messages)
    results = []
    async for chunk in gen:
        results.append(chunk)

    assert len(results) == 2
    assert results[0].delta == "Hello"
    assert results[1].delta == " world"
    assert results[1].message.content == "Hello world"


# ===================================================================
# 16. astream_chat() fallback to sync
# ===================================================================


@pytest.mark.asyncio
async def test_astream_chat_fallback_to_sync():
    """astream_chat() should fall back to sync stream_chat in a thread when astream is missing."""
    mock_llm = _make_chat_model()
    if hasattr(mock_llm, "astream"):
        del mock_llm.astream

    chunks = [
        _AIMessage(content="chunk1"),
        _AIMessage(content="chunk2"),
    ]

    def _stream(messages, **kwargs):
        for c in chunks:
            yield c

    mock_llm.stream = _stream

    adapter = _build_adapter(mock_llm)
    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]

    gen = await adapter.astream_chat(messages)
    results = []
    async for chunk in gen:
        results.append(chunk)

    assert len(results) == 2
    assert results[0].delta == "chunk1"


# ===================================================================
# 17. astream_complete() with native astream()
# ===================================================================


@pytest.mark.asyncio
async def test_astream_complete_native_astream():
    """astream_complete() should use native astream when available."""
    mock_llm = _make_plain_llm()

    chunks = [
        _AIMessage(content="part1"),
        _AIMessage(content="part2"),
    ]

    async def _astream(prompt, **kwargs):
        for c in chunks:
            yield c

    mock_llm.astream = _astream

    adapter = _build_adapter(mock_llm)

    gen = await adapter.astream_complete("prompt", formatted=True)
    results = []
    async for chunk in gen:
        results.append(chunk)

    assert len(results) == 2
    assert results[0].delta == "part1"
    assert results[1].delta == "part2"
    assert results[1].text == "part1part2"


@pytest.mark.asyncio
async def test_astream_complete_native_plain_string_chunks():
    """astream_complete() should handle plain string chunks from astream."""
    mock_llm = _make_plain_llm()

    async def _astream(prompt, **kwargs):
        for chunk in ["alpha", "beta"]:
            yield chunk

    mock_llm.astream = _astream

    adapter = _build_adapter(mock_llm)
    gen = await adapter.astream_complete("prompt", formatted=True)
    results = []
    async for chunk in gen:
        results.append(chunk)

    assert results[0].delta == "alpha"
    assert results[1].delta == "beta"
    assert results[1].text == "alphabeta"


# ===================================================================
# 18. astream_complete() fallback to sync
# ===================================================================


@pytest.mark.asyncio
async def test_astream_complete_fallback_to_sync():
    """astream_complete() should fall back to sync stream_complete via thread."""
    mock_llm = _make_plain_llm()
    if hasattr(mock_llm, "astream"):
        del mock_llm.astream

    chunks = [_AIMessage(content="f1"), _AIMessage(content="f2")]

    def _stream(prompt, **kwargs):
        for c in chunks:
            yield c

    mock_llm.stream = _stream

    adapter = _build_adapter(mock_llm)

    gen = await adapter.astream_complete("prompt", formatted=True)
    results = []
    async for chunk in gen:
        results.append(chunk)

    assert len(results) == 2
    assert results[0].delta == "f1"
    assert results[1].text == "f1f2"


# ===================================================================
# 19. is_chat_model() positive and negative
# ===================================================================


def test_is_chat_model_positive():
    """is_chat_model should return True for BaseChatModel instances."""
    mock = _make_chat_model()
    assert is_chat_model(mock) is True


def test_is_chat_model_negative():
    """is_chat_model should return False for non-BaseChatModel instances."""
    mock = _make_plain_llm()
    assert is_chat_model(mock) is False


# ===================================================================
# 20. to_lc_messages() for all roles
# ===================================================================


def test_to_lc_messages_user():
    """to_lc_messages should convert user messages to HumanMessage."""
    msgs = [ChatMessage(role=MessageRole.USER, content="Hello")]
    result = to_lc_messages(msgs)
    assert len(result) == 1
    assert isinstance(result[0], _HumanMessage)
    assert result[0].content == "Hello"


def test_to_lc_messages_assistant():
    """to_lc_messages should convert assistant messages to AIMessage."""
    msgs = [ChatMessage(role=MessageRole.ASSISTANT, content="Response")]
    result = to_lc_messages(msgs)
    assert len(result) == 1
    assert isinstance(result[0], _AIMessage)
    assert result[0].content == "Response"


def test_to_lc_messages_system():
    """to_lc_messages should convert system messages to SystemMessage."""
    msgs = [ChatMessage(role=MessageRole.SYSTEM, content="You are helpful")]
    result = to_lc_messages(msgs)
    assert len(result) == 1
    assert isinstance(result[0], _SystemMessage)
    assert result[0].content == "You are helpful"


def test_to_lc_messages_function():
    """to_lc_messages should convert function messages with name in additional_kwargs."""
    msgs = [
        ChatMessage(
            role=MessageRole.FUNCTION,
            content="result",
            additional_kwargs={"name": "my_func"},
        )
    ]
    result = to_lc_messages(msgs)
    assert len(result) == 1
    assert isinstance(result[0], _FunctionMessage)
    assert result[0].content == "result"
    assert result[0].name == "my_func"


def test_to_lc_messages_chatbot():
    """to_lc_messages should convert chatbot messages to LangChain ChatMessage."""
    msgs = [
        ChatMessage(
            role=MessageRole.CHATBOT,
            content="bot says",
            additional_kwargs={"role": "chatbot"},
        )
    ]
    result = to_lc_messages(msgs)
    assert len(result) == 1
    assert isinstance(result[0], _ChatMessage)


def test_to_lc_messages_invalid_role():
    """to_lc_messages should raise ValueError for unknown roles."""
    with pytest.raises((ValueError, KeyError)):
        class FakeMsg:
            role = "totally_invalid"
            content = "test"
            additional_kwargs = {}

        to_lc_messages([FakeMsg()])


def test_to_lc_messages_multiple():
    """to_lc_messages should handle a multi-message conversation."""
    msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content="Be helpful"),
        ChatMessage(role=MessageRole.USER, content="Hi"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hello!"),
    ]
    result = to_lc_messages(msgs)
    assert len(result) == 3
    assert isinstance(result[0], _SystemMessage)
    assert isinstance(result[1], _HumanMessage)
    assert isinstance(result[2], _AIMessage)


# ===================================================================
# 21. from_lc_messages() for all LangChain message types
# ===================================================================


def test_from_lc_messages_human():
    """from_lc_messages should convert HumanMessage to USER role."""
    lc_msgs = [_HumanMessage(content="user text")]
    result = from_lc_messages(lc_msgs)
    assert len(result) == 1
    assert result[0].role == MessageRole.USER
    assert result[0].content == "user text"


def test_from_lc_messages_ai():
    """from_lc_messages should convert AIMessage to ASSISTANT role."""
    lc_msgs = [_AIMessage(content="ai text")]
    result = from_lc_messages(lc_msgs)
    assert len(result) == 1
    assert result[0].role == MessageRole.ASSISTANT
    assert result[0].content == "ai text"


def test_from_lc_messages_system():
    """from_lc_messages should convert SystemMessage to SYSTEM role."""
    lc_msgs = [_SystemMessage(content="system text")]
    result = from_lc_messages(lc_msgs)
    assert len(result) == 1
    assert result[0].role == MessageRole.SYSTEM


def test_from_lc_messages_function():
    """from_lc_messages should convert FunctionMessage to FUNCTION role."""
    lc_msgs = [_FunctionMessage(content="func result", name="my_func")]
    result = from_lc_messages(lc_msgs)
    assert len(result) == 1
    assert result[0].role == MessageRole.FUNCTION


def test_from_lc_messages_chat():
    """from_lc_messages should convert LangChain ChatMessage to CHATBOT role."""
    lc_msgs = [_ChatMessage(content="chatbot text", role="chatbot")]
    result = from_lc_messages(lc_msgs)
    assert len(result) == 1
    assert result[0].role == MessageRole.CHATBOT


def test_from_lc_messages_unknown_type():
    """from_lc_messages should raise ValueError for unknown message types."""
    unknown = MagicMock()
    unknown.__class__ = type("UnknownMessage", (), {})
    unknown.content = "test"
    unknown.additional_kwargs = {}

    with pytest.raises(ValueError, match="Invalid message type"):
        from_lc_messages([unknown])


def test_from_lc_messages_roundtrip():
    """Messages should survive a to_lc -> from_lc roundtrip."""
    original = [
        ChatMessage(role=MessageRole.SYSTEM, content="sys prompt"),
        ChatMessage(role=MessageRole.USER, content="user query"),
        ChatMessage(role=MessageRole.ASSISTANT, content="assistant reply"),
    ]
    lc_msgs = to_lc_messages(original)
    roundtripped = from_lc_messages(lc_msgs)

    assert len(roundtripped) == 3
    assert roundtripped[0].role == MessageRole.SYSTEM
    assert roundtripped[0].content == "sys prompt"
    assert roundtripped[1].role == MessageRole.USER
    assert roundtripped[1].content == "user query"
    assert roundtripped[2].role == MessageRole.ASSISTANT
    assert roundtripped[2].content == "assistant reply"


# ===================================================================
# 22. _extract_model_name() with various attribute patterns
# ===================================================================


def test_extract_model_name_model_name_attr():
    """_extract_model_name should prefer model_name attribute."""
    llm = MagicMock()
    llm.model_name = "gpt-4"
    llm.model = "fallback"
    assert _extract_model_name(llm) == "gpt-4"


def test_extract_model_name_model_attr():
    """_extract_model_name should fall back to model attribute."""
    llm = MagicMock(spec=[])
    llm.model = "claude-3"
    assert _extract_model_name(llm) == "claude-3"


def test_extract_model_name_repo_id():
    """_extract_model_name should fall back to repo_id."""
    llm = MagicMock(spec=[])
    llm.repo_id = "meta-llama/Llama-2-7b"
    assert _extract_model_name(llm) == "meta-llama/Llama-2-7b"


def test_extract_model_name_model_id():
    """_extract_model_name should fall back to model_id."""
    llm = MagicMock(spec=[])
    llm.model_id = "my-custom-model"
    assert _extract_model_name(llm) == "my-custom-model"


def test_extract_model_name_class_name_fallback():
    """_extract_model_name should return class name when no known attrs exist."""
    llm = MagicMock(spec=[])
    result = _extract_model_name(llm)
    assert result == type(llm).__name__


def test_extract_model_name_skips_non_string():
    """_extract_model_name should skip non-string attribute values."""
    llm = MagicMock(spec=[])
    llm.model_name = 12345  # not a string
    llm.model = "valid-model"
    assert _extract_model_name(llm) == "valid-model"


def test_extract_model_name_skips_empty_string():
    """_extract_model_name should skip empty string values."""
    llm = MagicMock(spec=[])
    llm.model_name = ""
    llm.model = "fallback-model"
    assert _extract_model_name(llm) == "fallback-model"


# ===================================================================
# 23. _extract_max_tokens() with various attribute patterns
# ===================================================================


def test_extract_max_tokens_max_tokens():
    """_extract_max_tokens should prefer max_tokens."""
    llm = MagicMock(spec=[])
    llm.max_tokens = 1024
    assert _extract_max_tokens(llm) == 1024


def test_extract_max_tokens_max_new_tokens():
    """_extract_max_tokens should fall back to max_new_tokens."""
    llm = MagicMock(spec=[])
    llm.max_new_tokens = 512
    assert _extract_max_tokens(llm) == 512


def test_extract_max_tokens_maxTokens():
    """_extract_max_tokens should fall back to maxTokens (camelCase)."""
    llm = MagicMock(spec=[])
    llm.maxTokens = 256
    assert _extract_max_tokens(llm) == 256


def test_extract_max_tokens_max_output_tokens():
    """_extract_max_tokens should fall back to max_output_tokens."""
    llm = MagicMock(spec=[])
    llm.max_output_tokens = 2048
    assert _extract_max_tokens(llm) == 2048


def test_extract_max_tokens_default():
    """_extract_max_tokens should return -1 when no attrs found."""
    llm = MagicMock(spec=[])
    assert _extract_max_tokens(llm) == -1


def test_extract_max_tokens_skips_none():
    """_extract_max_tokens should skip None values."""
    llm = MagicMock(spec=[])
    llm.max_tokens = None
    llm.max_new_tokens = 128
    assert _extract_max_tokens(llm) == 128


def test_extract_max_tokens_skips_zero():
    """_extract_max_tokens should skip zero values."""
    llm = MagicMock(spec=[])
    llm.max_tokens = 0
    llm.max_new_tokens = 64
    assert _extract_max_tokens(llm) == 64


def test_extract_max_tokens_skips_negative():
    """_extract_max_tokens should skip negative values."""
    llm = MagicMock(spec=[])
    llm.max_tokens = -1
    llm.max_new_tokens = 100
    assert _extract_max_tokens(llm) == 100


# ===================================================================
# 24. _extract_context_window() with HuggingFace pipeline config
# ===================================================================


def test_extract_context_window_hf_pipeline_max_position_embeddings():
    """_extract_context_window should read max_position_embeddings from HF pipeline config."""
    config = MagicMock()
    config.to_dict.return_value = {"max_position_embeddings": 4096}

    model = MagicMock()
    model.config = config

    pipeline = MagicMock()
    pipeline.model = model

    llm = MagicMock(spec=[])
    llm.pipeline = pipeline
    assert _extract_context_window(llm) == 4096


def test_extract_context_window_hf_pipeline_n_positions():
    """_extract_context_window should read n_positions from HF pipeline config."""
    config = MagicMock()
    config.to_dict.return_value = {"n_positions": 2048}

    model = MagicMock()
    model.config = config

    pipeline = MagicMock()
    pipeline.model = model

    llm = MagicMock(spec=[])
    llm.pipeline = pipeline
    assert _extract_context_window(llm) == 2048


def test_extract_context_window_hf_pipeline_max_seq_len():
    """_extract_context_window should read max_seq_len from HF pipeline config."""
    config = MagicMock()
    config.to_dict.return_value = {"max_seq_len": 8192}

    model = MagicMock()
    model.config = config

    pipeline = MagicMock()
    pipeline.model = model

    llm = MagicMock(spec=[])
    llm.pipeline = pipeline
    assert _extract_context_window(llm) == 8192


def test_extract_context_window_hf_pipeline_priority():
    """max_position_embeddings should take priority over n_positions."""
    config = MagicMock()
    config.to_dict.return_value = {
        "max_position_embeddings": 4096,
        "n_positions": 2048,
    }

    model = MagicMock()
    model.config = config

    pipeline = MagicMock()
    pipeline.model = model

    llm = MagicMock(spec=[])
    llm.pipeline = pipeline
    assert _extract_context_window(llm) == 4096


# ===================================================================
# 25. _extract_context_window() with direct attributes
# ===================================================================


def test_extract_context_window_n_ctx():
    """_extract_context_window should read n_ctx (llama.cpp style)."""
    llm = MagicMock(spec=[])
    llm.n_ctx = 8192
    assert _extract_context_window(llm) == 8192


def test_extract_context_window_context_window_attr():
    """_extract_context_window should read context_window attribute."""
    llm = MagicMock(spec=[])
    llm.context_window = 16384
    assert _extract_context_window(llm) == 16384


def test_extract_context_window_max_seq_length():
    """_extract_context_window should read max_seq_length attribute."""
    llm = MagicMock(spec=[])
    llm.max_seq_length = 32768
    assert _extract_context_window(llm) == 32768


# ===================================================================
# 26. _extract_context_window() fallback
# ===================================================================


def test_extract_context_window_default():
    """_extract_context_window should return 3900 when no attrs are found."""
    llm = MagicMock(spec=[])
    assert _extract_context_window(llm) == 3900


def test_extract_context_window_skips_zero():
    """_extract_context_window should skip zero values for direct attrs."""
    llm = MagicMock(spec=[])
    llm.n_ctx = 0
    llm.context_window = 4096
    assert _extract_context_window(llm) == 4096


# ===================================================================
# 27. get_llm_metadata() generic fallback path
# ===================================================================


def test_get_llm_metadata_generic_plain():
    """get_llm_metadata on a generic model should extract metadata via helper functions."""
    llm = _make_plain_llm()
    llm.model_name = "my-model"
    llm.max_tokens = 512
    llm.n_ctx = 4096

    meta = get_llm_metadata(llm)

    assert meta.model_name == "my-model"
    assert meta.num_output == 512
    assert meta.context_window == 4096
    assert meta.is_chat_model is False


def test_get_llm_metadata_generic_chat_model():
    """get_llm_metadata on a generic chat model should set is_chat_model=True."""
    llm = _make_chat_model()
    llm.model_name = "chat-model"
    llm.max_tokens = 256
    llm.context_window = 8192

    meta = get_llm_metadata(llm)

    assert meta.model_name == "chat-model"
    assert meta.num_output == 256
    assert meta.context_window == 8192
    assert meta.is_chat_model is True


def test_get_llm_metadata_generic_defaults():
    """get_llm_metadata with no known attrs should use sensible defaults."""
    llm = _make_plain_llm()
    llm.model_name = None
    llm.model = None
    llm.max_tokens = None

    meta = get_llm_metadata(llm)

    assert meta.context_window == 3900
    assert meta.num_output == -1
    assert isinstance(meta.model_name, str)
    assert len(meta.model_name) > 0


# ===================================================================
# Additional edge-case / integration tests
# ===================================================================


def test_class_name():
    """LangChainLLM.class_name() should return the expected string."""
    assert LangChainLLM.class_name() == "LangChainLLM"


def test_llm_property():
    """The llm property should return the underlying LangChain model."""
    mock_llm = _make_plain_llm()
    adapter = _build_adapter(mock_llm)
    assert adapter.llm is mock_llm


def test_chat_preserves_additional_kwargs_on_invoke():
    """chat() should pass extra kwargs through to invoke."""
    mock_llm = _make_chat_model()
    returned_ai = _AIMessage(content="ok")
    mock_llm.invoke = MagicMock(return_value=returned_ai)

    adapter = _build_adapter(mock_llm)
    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]

    adapter.chat(messages, temperature=0.5)

    _, kwargs = mock_llm.invoke.call_args
    assert kwargs.get("temperature") == 0.5


def test_complete_uses_completion_to_prompt_when_not_formatted():
    """complete() should call completion_to_prompt when formatted=False."""
    mock_llm = _make_plain_llm()
    mock_llm.invoke = MagicMock(return_value="result")

    adapter = _build_adapter(mock_llm)
    adapter.completion_to_prompt = lambda p: f"<prompt>{p}</prompt>"

    response = adapter.complete("raw prompt", formatted=False)

    assert isinstance(response, CompletionResponse)
    # The invoke should have received the formatted prompt
    called_with = mock_llm.invoke.call_args[0][0]
    assert called_with == "<prompt>raw prompt</prompt>"
