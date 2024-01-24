"""ReAct chat engine."""

import logging
from threading import Thread
from typing import Any, List, Optional, Type, cast, Tuple

from llama_index.llms.base import ChatMessage, ChatResponse
from llama_index.callbacks import CallbackManager, trace_method
from llama_index.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
)
from llama_index.chat_engine.utils import response_gen_from_query_engine
from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.core.llms.types import ChatMessage, MessageRole
from llama_index.core.response.schema import RESPONSE_TYPE, StreamingResponse
from llama_index.llm_predictor.base import LLMPredictorType
from llama_index.llms.generic_utils import messages_to_history_str
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.service_context import ServiceContext
from llama_index.tools import ToolOutput
from llama_index.chat_engine.react.prompts import REACT_CHAT_ENGINE_SYSTEM_HEADER
from llama_index.chat_engine.react.formatter import ReActChatEngineFormatter
from llama_index.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.utils import print_text, unit_generator

# TODO: migrate
from llama_index.agent.react.output_parser import ReActOutputParser
from llama_index.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)

logger = logging.getLogger(__name__)


class ReActChatEngine(BaseChatEngine):
    """ReAct Chat Engine.

    Args:
        query_engine (BaseQueryEngine): search tool to use
        react_prompt (BasePromptTemplate): ReAct prompt template
        memory (BaseMemory): chat memory
        llm (LLMPredictorType): language model
        verbose (bool): verbose
        callback_manager (Optional[CallbackManager]): callback manager
    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        react_chat_formatter: ReActChatEngineFormatter,
        output_parser: ReActOutputParser,
        memory: BaseMemory,
        llm: LLMPredictorType,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._query_engine = query_engine
        self._react_chat_formatter = react_chat_formatter
        self._output_parser = output_parser
        self._memory = memory
        self._llm = llm
        self._verbose = verbose
        self.callback_manager = callback_manager or CallbackManager([])

    @classmethod
    def from_defaults(
        cls,
        query_engine: BaseQueryEngine,
        react_chat_formatter: Optional[ReActChatEngineFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        system_prompt: str = REACT_CHAT_ENGINE_SYSTEM_HEADER,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        service_context: Optional[ServiceContext] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "ReActChatEngine":
        """Initialize a ReActChatEngine from default parameters."""
        react_chat_formatter = react_chat_formatter or ReActChatEngineFormatter(
            system_header=system_prompt
        )
        output_parser = output_parser or ReActOutputParser()

        service_context = service_context or ServiceContext.from_defaults()
        llm = service_context.llm

        chat_history = chat_history or []
        memory = memory or memory_cls.from_defaults(chat_history=chat_history, llm=llm)

        return cls(
            query_engine,
            react_chat_formatter,
            memory,
            llm,
            verbose=verbose,
            callback_manager=service_context.callback_manager,
        )

    def _condense_question(
        self, chat_history: List[ChatMessage], last_message: str
    ) -> str:
        """
        Generate standalone question from conversation context and last message.
        """
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
        if isinstance(response, StreamingResponse):
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

    def _extract_reasoning_step(
        self, output: ChatResponse, is_streaming: bool = False
    ) -> Tuple[str, List[BaseReasoningStep], bool]:
        """
        Extracts the reasoning step from the given output.

        This method parses the message content from the output,
        extracts the reasoning step, and determines whether the processing is
        complete. It also performs validation checks on the output and
        handles possible errors.
        """
        if output.message.content is None:
            raise ValueError("Got empty message.")
        message_content = output.message.content
        current_reasoning = []
        try:
            reasoning_step = self._output_parser.parse(message_content, is_streaming)
        except BaseException as exc:
            raise ValueError(f"Could not parse output: {message_content}") from exc
        if self._verbose:
            print_text(f"{reasoning_step.get_content()}\n", color="pink")
        current_reasoning.append(reasoning_step)

        if reasoning_step.is_done:
            return message_content, current_reasoning, True

        reasoning_step = cast(ActionReasoningStep, reasoning_step)
        if not isinstance(reasoning_step, ActionReasoningStep):
            raise ValueError(f"Expected ActionReasoningStep, got {reasoning_step}")

        return message_content, current_reasoning, False

    def _run_step(
        self,
        message: str,
        chat_history: List[ChatMessage],
    ) -> Any:
        """Run a single reasoning step."""
        input_chat = self._react_chat_formatter.format(
            data_desc=self.data_desc,
        )
        chat_response = self._llm.chat(input_chat)
        

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        chat_history = chat_history or self._memory.get()

        # Generate standalone question from conversation context and last message
        condensed_question = self._condense_question(chat_history, message)

        log_str = f"Querying with: {condensed_question}"
        logger.info(log_str)
        if self._verbose:
            print(log_str)

        # TODO: right now, query engine uses class attribute to configure streaming,
        #       we are moving towards separate streaming and non-streaming methods.
        #       In the meanwhile, use this hack to toggle streaming.
        from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

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
        chat_history = chat_history or self._memory.get()

        # Generate standalone question from conversation context and last message
        condensed_question = self._condense_question(chat_history, message)

        log_str = f"Querying with: {condensed_question}"
        logger.info(log_str)
        if self._verbose:
            print(log_str)

        # TODO: right now, query engine uses class attribute to configure streaming,
        #       we are moving towards separate streaming and non-streaming methods.
        #       In the meanwhile, use this hack to toggle streaming.
        from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

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
                target=response.write_response_to_history, args=(self._memory,)
            )
            thread.start()
        else:
            raise ValueError("Streaming is not enabled. Please use chat() instead.")
        return response

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        chat_history = chat_history or self._memory.get()

        # Generate standalone question from conversation context and last message
        condensed_question = await self._acondense_question(chat_history, message)

        log_str = f"Querying with: {condensed_question}"
        logger.info(log_str)
        if self._verbose:
            print(log_str)

        # TODO: right now, query engine uses class attribute to configure streaming,
        #       we are moving towards separate streaming and non-streaming methods.
        #       In the meanwhile, use this hack to toggle streaming.
        from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

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
        self._memory.put(ChatMessage(role=MessageRole.USER, content=message))
        self._memory.put(
            ChatMessage(role=MessageRole.ASSISTANT, content=str(query_response))
        )

        return AgentChatResponse(response=str(query_response), sources=[tool_output])

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        chat_history = chat_history or self._memory.get()

        # Generate standalone question from conversation context and last message
        condensed_question = await self._acondense_question(chat_history, message)

        log_str = f"Querying with: {condensed_question}"
        logger.info(log_str)
        if self._verbose:
            print(log_str)

        # TODO: right now, query engine uses class attribute to configure streaming,
        #       we are moving towards separate streaming and non-streaming methods.
        #       In the meanwhile, use this hack to toggle streaming.
        from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

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
        if (
            isinstance(query_response, StreamingResponse)
            and query_response.response_gen is not None
        ):
            # override the generator to include writing to chat history
            # TODO: query engine does not support async generator yet
            self._memory.put(ChatMessage(role=MessageRole.USER, content=message))
            response = StreamingAgentChatResponse(
                chat_stream=response_gen_from_query_engine(query_response.response_gen),
                sources=[tool_output],
            )
            thread = Thread(
                target=response.write_response_to_history, args=(self._memory,)
            )
            thread.start()
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
