import asyncio
import logging
from typing import Any, List, Optional, Type

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    StreamingResponse,
    AsyncStreamingResponse,
)
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
)
from llama_index.core.chat_engine.utils import (
    response_gen_from_query_engine,
    aresponse_gen_from_query_engine,
)
from llama_index.core.base.llms.generic_utils import messages_to_history_str
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.settings import Settings

from llama_index.core.tools import ToolOutput
from llama_index.core.types import Thread

logger = logging.getLogger(__name__)


DEFAULT_TEMPLATE = """\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""

DEFAULT_PROMPT = PromptTemplate(DEFAULT_TEMPLATE)


class CondenseQuestionChatEngine(BaseChatEngine):
    """
    Condense Question Chat Engine.

    First generate a standalone question from conversation context and last message,
    then query the query engine for a response.
    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        condense_question_prompt: BasePromptTemplate,
        memory: BaseMemory,
        llm: LLM,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._query_engine = query_engine
        self._condense_question_prompt = condense_question_prompt
        self._memory = memory
        self._llm = llm
        self._verbose = verbose
        self.callback_manager = callback_manager or CallbackManager([])

    @classmethod
    def from_defaults(
        cls,
        query_engine: BaseQueryEngine,
        condense_question_prompt: Optional[BasePromptTemplate] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        verbose: bool = False,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> "CondenseQuestionChatEngine":
        """Initialize a CondenseQuestionChatEngine from default parameters."""
        condense_question_prompt = condense_question_prompt or DEFAULT_PROMPT

        llm = llm or Settings.llm

        chat_history = chat_history or []
        memory = memory or memory_cls.from_defaults(chat_history=chat_history, llm=llm)

        if system_prompt is not None:
            raise NotImplementedError(
                "system_prompt is not supported for CondenseQuestionChatEngine."
            )
        if prefix_messages is not None:
            raise NotImplementedError(
                "prefix_messages is not supported for CondenseQuestionChatEngine."
            )

        return cls(
            query_engine,
            condense_question_prompt,
            memory,
            llm,
            verbose=verbose,
            callback_manager=Settings.callback_manager,
        )

    def _condense_question(
        self, chat_history: List[ChatMessage], last_message: str
    ) -> str:
        """
        Generate standalone question from conversation context and last message.
        """
        if not chat_history:
            # Keep the question as is if there's no conversation context.
            return last_message

        chat_history_str = messages_to_history_str(chat_history)
        logger.debug(chat_history_str)

        return self._llm.predict(
            self._condense_question_prompt,
            question=last_message,
            chat_history=chat_history_str,
        )

    async def _acondense_question(
        self, chat_history: List[ChatMessage], last_message: str
    ) -> str:
        """
        Generate standalone question from conversation context and last message.
        """
        if not chat_history:
            # Keep the question as is if there's no conversation context.
            return last_message

        chat_history_str = messages_to_history_str(chat_history)
        logger.debug(chat_history_str)

        return await self._llm.apredict(
            self._condense_question_prompt,
            question=last_message,
            chat_history=chat_history_str,
        )

    def _get_tool_output_from_response(
        self, query: str, response: RESPONSE_TYPE
    ) -> ToolOutput:
        if isinstance(response, (StreamingResponse, AsyncStreamingResponse)):
            return ToolOutput(
                content="",
                tool_name="query_engine",
                raw_input={"query": query},
                raw_output=response,
            )
        else:
            return ToolOutput(
                content=str(response),
                tool_name="query_engine",
                raw_input={"query": query},
                raw_output=response,
            )

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        chat_history = chat_history or self._memory.get(input=message)

        # Generate standalone question from conversation context and last message
        condensed_question = self._condense_question(chat_history, message)

        log_str = f"Querying with: {condensed_question}"
        logger.info(log_str)
        if self._verbose:
            print(log_str)

        # TODO: right now, query engine uses class attribute to configure streaming,
        #       we are moving towards separate streaming and non-streaming methods.
        #       In the meanwhile, use this hack to toggle streaming.
        from llama_index.core.query_engine.retriever_query_engine import (
            RetrieverQueryEngine,
        )

        if isinstance(self._query_engine, RetrieverQueryEngine):
            is_streaming = self._query_engine._response_synthesizer._streaming
            self._query_engine._response_synthesizer._streaming = False

        # Query with standalone question
        query_response = self._query_engine.query(condensed_question)

        # NOTE: reset streaming flag
        if isinstance(self._query_engine, RetrieverQueryEngine):
            self._query_engine._response_synthesizer._streaming = is_streaming

        tool_output = self._get_tool_output_from_response(
            condensed_question, query_response
        )

        # Record response
        self._memory.put(ChatMessage(role=MessageRole.USER, content=message))
        self._memory.put(
            ChatMessage(role=MessageRole.ASSISTANT, content=str(query_response))
        )

        return AgentChatResponse(response=str(query_response), sources=[tool_output])

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        chat_history = chat_history or self._memory.get(input=message)

        # Generate standalone question from conversation context and last message
        condensed_question = self._condense_question(chat_history, message)

        log_str = f"Querying with: {condensed_question}"
        logger.info(log_str)
        if self._verbose:
            print(log_str)

        # TODO: right now, query engine uses class attribute to configure streaming,
        #       we are moving towards separate streaming and non-streaming methods.
        #       In the meanwhile, use this hack to toggle streaming.
        from llama_index.core.query_engine.retriever_query_engine import (
            RetrieverQueryEngine,
        )

        if isinstance(self._query_engine, RetrieverQueryEngine):
            is_streaming = self._query_engine._response_synthesizer._streaming
            self._query_engine._response_synthesizer._streaming = True

        # Query with standalone question
        query_response = self._query_engine.query(condensed_question)

        # NOTE: reset streaming flag
        if isinstance(self._query_engine, RetrieverQueryEngine):
            self._query_engine._response_synthesizer._streaming = is_streaming

        tool_output = self._get_tool_output_from_response(
            condensed_question, query_response
        )

        # Record response
        if (
            isinstance(query_response, StreamingResponse)
            and query_response.response_gen is not None
        ):
            # override the generator to include writing to chat history
            self._memory.put(ChatMessage(role=MessageRole.USER, content=message))
            response = StreamingAgentChatResponse(
                chat_stream=response_gen_from_query_engine(query_response.response_gen),
                sources=[tool_output],
            )
            thread = Thread(
                target=response.write_response_to_history,
                args=(self._memory,),
            )
            thread.start()
        else:
            raise ValueError("Streaming is not enabled. Please use chat() instead.")
        return response

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        chat_history = chat_history or await self._memory.aget(input=message)

        # Generate standalone question from conversation context and last message
        condensed_question = await self._acondense_question(chat_history, message)

        log_str = f"Querying with: {condensed_question}"
        logger.info(log_str)
        if self._verbose:
            print(log_str)

        # TODO: right now, query engine uses class attribute to configure streaming,
        #       we are moving towards separate streaming and non-streaming methods.
        #       In the meanwhile, use this hack to toggle streaming.
        from llama_index.core.query_engine.retriever_query_engine import (
            RetrieverQueryEngine,
        )

        if isinstance(self._query_engine, RetrieverQueryEngine):
            is_streaming = self._query_engine._response_synthesizer._streaming
            self._query_engine._response_synthesizer._streaming = False

        # Query with standalone question
        query_response = await self._query_engine.aquery(condensed_question)

        # NOTE: reset streaming flag
        if isinstance(self._query_engine, RetrieverQueryEngine):
            self._query_engine._response_synthesizer._streaming = is_streaming

        tool_output = self._get_tool_output_from_response(
            condensed_question, query_response
        )

        # Record response
        await self._memory.aput(ChatMessage(role=MessageRole.USER, content=message))
        await self._memory.aput(
            ChatMessage(role=MessageRole.ASSISTANT, content=str(query_response))
        )

        return AgentChatResponse(response=str(query_response), sources=[tool_output])

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        chat_history = chat_history or await self._memory.aget(input=message)

        # Generate standalone question from conversation context and last message
        condensed_question = await self._acondense_question(chat_history, message)

        log_str = f"Querying with: {condensed_question}"
        logger.info(log_str)
        if self._verbose:
            print(log_str)

        # TODO: right now, query engine uses class attribute to configure streaming,
        #       we are moving towards separate streaming and non-streaming methods.
        #       In the meanwhile, use this hack to toggle streaming.
        from llama_index.core.query_engine.retriever_query_engine import (
            RetrieverQueryEngine,
        )

        if isinstance(self._query_engine, RetrieverQueryEngine):
            is_streaming = self._query_engine._response_synthesizer._streaming
            self._query_engine._response_synthesizer._streaming = True

        # Query with standalone question
        query_response = await self._query_engine.aquery(condensed_question)

        # NOTE: reset streaming flag
        if isinstance(self._query_engine, RetrieverQueryEngine):
            self._query_engine._response_synthesizer._streaming = is_streaming

        tool_output = self._get_tool_output_from_response(
            condensed_question, query_response
        )

        # Record response
        if isinstance(query_response, AsyncStreamingResponse):
            # override the generator to include writing to chat history
            # TODO: query engine does not support async generator yet
            await self._memory.aput(ChatMessage(role=MessageRole.USER, content=message))
            response = StreamingAgentChatResponse(
                achat_stream=aresponse_gen_from_query_engine(
                    query_response.async_response_gen()
                ),
                sources=[tool_output],
            )
            response.awrite_response_to_history_task = asyncio.create_task(
                response.awrite_response_to_history(self._memory)
            )

        else:
            raise ValueError("Streaming is not enabled. Please use achat() instead.")
        return response

    def reset(self) -> None:
        # Clear chat history
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self._memory.get_all()
