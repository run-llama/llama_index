import asyncio
from typing import Any, List, Union
from unittest.mock import Mock
import pytest

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.response.schema import Response, StreamingResponse
from llama_index.core.chat_engine.condense_question import (
    CondenseQuestionChatEngine,
)
from llama_index.core.llms.mock import MockLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


def test_condense_question_chat_engine(patch_llm_predictor) -> None:
    query_engine = Mock(spec=BaseQueryEngine)
    query_engine.query.side_effect = lambda x: Response(response=x)
    engine = CondenseQuestionChatEngine.from_defaults(query_engine=query_engine)

    engine.reset()
    response = engine.chat("Test message 1")
    assert str(response) == "Test message 1"

    response = engine.chat("Test message 2")
    assert str(response) == (
        "{"
        "'question': 'Test message 2', "
        "'chat_history': 'user: Test message 1\\nassistant: Test message 1'"
        "}"
    )

    engine.reset()
    response = engine.chat("Test message 3")
    assert str(response) == "Test message 3"


def test_condense_question_chat_engine_with_init_history(patch_llm_predictor) -> None:
    query_engine = Mock(spec=BaseQueryEngine)
    query_engine.query.side_effect = lambda x: Response(response=x)
    engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        chat_history=[
            ChatMessage(role=MessageRole.USER, content="test human message"),
            ChatMessage(role=MessageRole.ASSISTANT, content="test ai message"),
        ],
    )

    response = engine.chat("new human message")
    assert str(response) == (
        "{'question': 'new human message', 'chat_history': 'user: test human "
        "message\\nassistant: test ai message'}"
    )


def test_stream_chat_history_write_completes_on_early_exit() -> None:
    def token_gen():
        for token in ["Hello", " ", "World", "!"]:
            yield token

    query_engine = Mock(spec=BaseQueryEngine)
    query_engine.query.return_value = StreamingResponse(response_gen=token_gen())
    engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine, llm=MockLLM()
    )
    engine._memory = ChatMemoryBuffer.from_defaults()

    response = engine.stream_chat("Hello!")
    gen = response.response_gen
    for i, _ in enumerate(gen):
        if i >= 2:
            break
    gen.close()
    assert len(engine.chat_history) == 2
    assert engine.chat_history[0].role == MessageRole.USER
    assert engine.chat_history[1].role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_condense_question_concurrent_achat_and_astream_chat() -> None:
    class _SlowAsyncRetriever(BaseRetriever):
        def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            return [NodeWithScore(node=TextNode(text="context"))]

        async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            await asyncio.sleep(0.02)
            return [NodeWithScore(node=TextNode(text="context"))]

    llm = MockLLM()
    for initial_streaming in (False, True):
        query_engine = RetrieverQueryEngine.from_args(
            retriever=_SlowAsyncRetriever(), llm=llm, streaming=initial_streaming
        )
        engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine, llm=llm
        )

        if not initial_streaming:
            first_task = asyncio.create_task(engine.astream_chat("question 1"))
            await asyncio.sleep(0.005)
            second_task = asyncio.create_task(engine.achat("question 2"))
            stream_resp, chat_resp = await asyncio.gather(first_task, second_task)
        else:
            first_task = asyncio.create_task(engine.achat("question 1"))
            await asyncio.sleep(0.005)
            second_task = asyncio.create_task(engine.astream_chat("question 2"))
            chat_resp, stream_resp = await asyncio.gather(first_task, second_task)

        chunks = [chunk async for chunk in stream_resp.async_response_gen()]
        assert len(chunks) > 0
        assert isinstance(chat_resp.response, str) and len(chat_resp.response) > 0
        assert query_engine._response_synthesizer._streaming is initial_streaming


def test_condense_question_custom_synthesizer_streaming_toggle() -> None:
    from llama_index.core.callbacks import CallbackManager

    class _SimpleRetriever(BaseRetriever):
        def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            return [NodeWithScore(node=TextNode(text="context"))]

    class _DuckSynthesizer:
        def __init__(self) -> None:
            self._streaming = False
            self.callback_manager = CallbackManager([])

        def synthesize(
            self, query: Any, nodes: List[NodeWithScore], **kwargs: Any
        ) -> Union[Response, StreamingResponse]:
            if self._streaming:
                return StreamingResponse(response_gen=(t for t in ["hello"]))
            return Response("world")

    synthesizer = _DuckSynthesizer()
    query_engine = RetrieverQueryEngine(
        retriever=_SimpleRetriever(),
        response_synthesizer=synthesizer,  # type: ignore[arg-type]
    )
    engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine, llm=MockLLM()
    )

    stream_resp = engine.stream_chat("q")
    assert "".join(stream_resp.response_gen) == "hello"
    assert synthesizer._streaming is False
