import os
import time
import typing
from uuid import uuid4

import pytest

from llama_index.callbacks.langfuse import LangfuseSpanHandler
from llama_index.core import PromptTemplate, Settings
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.llms.openai import OpenAI

try:
    import pydantic.v1 as pydantic  # type: ignore
except ImportError:
    import pydantic  # type: ignore

from langfuse.api.client import FernLangfuse

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)


skip_cond = pytest.mark.skipif(
    any(
        var not in os.environ
        for var in [
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_HOST",
            "OPENAI_API_KEY",
        ]
    ),
    reason="Please set your Langfuse keys and OpenAI API Key.",
)


def create_uuid():
    return str(uuid4())


def get_api():
    return FernLangfuse(
        username=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        password=os.environ.get("LANGFUSE_SECRET_KEY"),
        base_url=os.environ.get("LANGFUSE_HOST"),
    )


class LlmUsageWithCost(pydantic.BaseModel):
    prompt_tokens: typing.Optional[int] = pydantic.Field(
        alias="promptTokens", default=None
    )
    completion_tokens: typing.Optional[int] = pydantic.Field(
        alias="completionTokens", default=None
    )
    total_tokens: typing.Optional[int] = pydantic.Field(
        alias="totalTokens", default=None
    )
    input_cost: typing.Optional[float] = pydantic.Field(alias="inputCost", default=None)
    output_cost: typing.Optional[float] = pydantic.Field(
        alias="outputCost", default=None
    )
    total_cost: typing.Optional[float] = pydantic.Field(alias="totalCost", default=None)


class CompletionUsage(pydantic.BaseModel):
    completion_tokens: int
    """Number of tokens in the generated completion."""

    prompt_tokens: int
    """Number of tokens in the prompt."""

    total_tokens: int
    """Total number of tokens used in the request (prompt + completion)."""


class LlmUsage(pydantic.BaseModel):
    prompt_tokens: typing.Optional[int] = pydantic.Field(
        alias="promptTokens", default=None
    )
    completion_tokens: typing.Optional[int] = pydantic.Field(
        alias="completionTokens", default=None
    )
    total_tokens: typing.Optional[int] = pydantic.Field(
        alias="totalTokens", default=None
    )

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {
            "by_alias": True,
            "exclude_unset": True,
            **kwargs,
        }
        return super().model_dump_json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults: typing.Any = {
            "by_alias": True,
            "exclude_unset": True,
            **kwargs,
        }
        return super().model_dump(**kwargs_with_defaults)


def get_llama_index_index(callback, force_rebuild: bool = False):
    dispatcher = get_dispatcher()
    dispatcher.span_handlers.clear()
    dispatcher.add_span_handler(callback)

    PERSIST_DIR = "tests/mocks/llama-index-storage"

    if not os.path.exists(PERSIST_DIR) or force_rebuild:
        print("Building RAG index...")
        documents = SimpleDirectoryReader(
            "static", ["static/state_of_the_union_short.txt"]
        ).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        print("Using pre-built index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    return index


def validate_embedding_generation(generation):
    return all(
        [
            generation.name == "OpenAIEmbedding",
            generation.usage.input > 0,
            generation.usage.output > 0,
            generation.usage.total > 0,  # For embeddings, only total tokens are logged
            bool(generation.input),
            bool(generation.output),
        ]
    )


def validate_llm_generation(generation, model_name="OpenAI"):
    return all(
        [
            generation.name == model_name,
            generation.usage.input > 0,
            generation.usage.output > 0,
            generation.usage.total > 0,
            bool(generation.input),
            bool(generation.output),
        ]
    )


@skip_cond
def test_callback_init():
    callback = LangfuseSpanHandler(
        release="release",
        version="version",
        session_id="session-id",
        user_id="user-id",
        metadata={"key": "value"},
        tags=["tag1", "tag2"],
    )

    assert callback._handler.langfuse.release == "release"
    assert callback._handler.session_id == "session-id"
    assert callback._handler.user_id == "user-id"
    assert callback._handler.metadata == {"key": "value"}
    assert callback._handler.tags == ["tag1", "tag2"]
    assert callback._handler.version == "version"
    assert callback._handler._task_manager is not None


@skip_cond
def test_constructor_kwargs():
    callback = LangfuseSpanHandler(
        release="release",
        version="version",
        session_id="session-id",
        user_id="user-id",
        metadata={"key": "value"},
        tags=["tag1", "tag2"],
    )
    get_llama_index_index(callback, force_rebuild=True)
    assert callback._trace is not None

    trace_id = callback._trace.id
    assert trace_id is not None

    callback.flush()
    time.sleep(5)
    trace_data = get_api().trace.get(trace_id)
    assert trace_data is not None

    assert trace_data.release == "release"
    assert trace_data.version == "version"
    assert trace_data.session_id == "session-id"
    assert trace_data.user_id == "user-id"
    assert "key" in trace_data.metadata and trace_data.metadata["key"] == "value"
    assert trace_data.tags == ["tag1", "tag2"]


@skip_cond
def test_callback_from_index_construction():
    callback = LangfuseSpanHandler()
    get_llama_index_index(callback, force_rebuild=True)

    assert callback._trace is not None

    trace_id = callback._trace.id
    assert trace_id is not None

    callback.flush()
    time.sleep(5)
    trace_data = get_api().trace.get(trace_id)
    assert trace_data is not None

    observations = trace_data.observations

    assert any(o.name == "OpenAIEmbedding" for o in observations)

    # Test embedding generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    assert len(generations) == 1  # Only one generation event for all embedded chunks

    generation = generations[0]
    assert validate_embedding_generation(generation)


@skip_cond
def test_callback_from_query_engine():
    callback = LangfuseSpanHandler()
    index = get_llama_index_index(callback)
    index.as_query_engine().query(
        "What did the speaker achieve in the past twelve months?"
    )

    assert callback._trace is not None

    callback.flush()
    time.sleep(5)
    trace_data = get_api().trace.get(callback._trace.id)

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    assert len(generations) > 1

    embedding_generation, gen_2, llm_generation, gen_4 = generations
    assert validate_embedding_generation(embedding_generation)
    assert validate_embedding_generation(gen_2)
    assert validate_llm_generation(llm_generation)
    assert validate_llm_generation(gen_4)


@skip_cond
def test_callback_from_chat_engine():
    callback = LangfuseSpanHandler()
    index = get_llama_index_index(callback)
    index.as_chat_engine().chat(
        "What did the speaker achieve in the past twelve months?"
    )

    callback.flush()
    time.sleep(5)
    trace_data = get_api().trace.get(callback._trace.id)

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    embedding_generations = [g for g in generations if g.name == "OpenAIEmbedding"]
    llm_generations = [g for g in generations if g.name == "OpenAI"]

    assert len(embedding_generations) > 0
    assert len(llm_generations) > 0

    assert all(validate_embedding_generation(g) for g in embedding_generations)
    assert all(validate_llm_generation(g) for g in llm_generations)


@skip_cond
def test_callback_from_query_engine_stream():
    callback = LangfuseSpanHandler()
    index = get_llama_index_index(callback)
    stream_response = index.as_query_engine(streaming=True).query(
        "What did the speaker achieve in the past twelve months?"
    )

    for token in stream_response.response_gen:
        print(token, end="")

    callback.flush()
    time.sleep(5)
    trace_data = get_api().trace.get(callback._trace.id)

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    embedding_generations = [g for g in generations if g.name == "OpenAIEmbedding"]
    llm_generations = [g for g in generations if g.name == "OpenAI"]

    assert len(embedding_generations) > 0
    assert len(llm_generations) > 0

    assert all(validate_embedding_generation(g) for g in embedding_generations)


@skip_cond
def test_callback_from_chat_stream():
    callback = LangfuseSpanHandler()
    index = get_llama_index_index(callback)
    stream_response = index.as_chat_engine().stream_chat(
        "What did the speaker achieve in the past twelve months?"
    )

    for token in stream_response.response_gen:
        print(token, end="")

    time.sleep(10)
    callback.flush()
    trace_data = get_api().trace.get(callback._trace.id)

    # Test LLM generation
    generations = sorted(
        [o for o in trace_data.observations if o.type == "GENERATION"],
        key=lambda o: o.start_time,
    )
    embedding_generations = [g for g in generations if g.name == "OpenAIEmbedding"]
    llm_generations = [g for g in generations if g.name == "OpenAI"]

    assert len(embedding_generations) > 0
    assert len(llm_generations) > 0

    assert all(validate_embedding_generation(g) for g in embedding_generations)
    assert any(
        validate_llm_generation(g) for g in llm_generations
    )  # just need one LLM generation that contains output


@skip_cond
def test_callback_from_query_pipeline():
    callback = LangfuseSpanHandler()
    dispatcher = get_dispatcher()
    dispatcher.add_span_handler(callback)

    prompt_str = "Please generate related movies to {movie_name}"
    prompt_tmpl = PromptTemplate(prompt_str)
    models = [
        ("OpenAI", OpenAI(model="gpt-3.5-turbo")),
    ]

    for model_name, llm in models:
        pipeline = QueryPipeline(
            chain=[prompt_tmpl, llm],
            verbose=True,
            callback_manager=Settings.callback_manager,
        )
        pipeline.run(movie_name="The Matrix")

        callback.flush()
        time.sleep(5)
        trace_data = get_api().trace.get(callback._trace.id)
        observations = trace_data.observations
        llm_generations = list(
            filter(
                lambda o: o.type == "GENERATION" and o.name == model_name,
                observations,
            )
        )

        assert len(llm_generations) > 0
        assert validate_llm_generation(llm_generations[0], model_name=model_name)


@skip_cond
def test_disabled_langfuse():
    callback = LangfuseSpanHandler(enabled=False)
    get_llama_index_index(callback, force_rebuild=True)

    assert callback._trace is not None

    trace_id = callback._trace.id
    assert trace_id is not None

    assert callback._handler.langfuse.task_manager._queue.empty()

    callback.flush()
    time.sleep(1)

    with pytest.raises(Exception):
        get_api().trace.get(trace_id)
