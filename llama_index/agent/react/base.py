# ReAct agent

import asyncio
from threading import Thread
from typing import Any, List, Optional, Sequence, Tuple, Type, cast

from llama_index.agent.react.formatter import ReActChatFormatter
from llama_index.agent.react.output_parser import ReActOutputParser
from llama_index.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.agent.types import BaseAgent
from llama_index.bridge.langchain import print_text
from llama_index.callbacks import (
    CallbackManager,
    trace_method,
    EventPayload,
    CBEventType,
)
from llama_index.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.llms.base import LLM, ChatMessage, ChatResponse, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.memory.types import BaseMemory
from llama_index.tools import BaseTool, adapt_to_async_tool

DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"


class ReActAgent(BaseAgent):
    """ReAct agent.

    Uses a ReAct prompt that can be used in both chat and text
    completion endpoints.

    Can take in a set of tools that require structured inputs.

    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        memory: BaseMemory,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        self._llm = llm
        self._tools = [adapt_to_async_tool(tool) for tool in tools]
        self._tools_dict = {tool.metadata.name: tool for tool in self._tools}
        self._memory = memory
        self._max_iterations = max_iterations
        self._react_chat_formatter = react_chat_formatter or ReActChatFormatter(
            tools=tools
        )
        self._output_parser = output_parser or ReActOutputParser()
        self.callback_manager = callback_manager or CallbackManager([])
        self._verbose = verbose

    @classmethod
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "ReActAgent":
        tools = tools or []
        chat_history = chat_history or []
        llm = llm or OpenAI(model=DEFAULT_MODEL_NAME)
        if callback_manager is not None:
            llm.callback_manager = callback_manager

        memory = memory or memory_cls.from_defaults(chat_history=chat_history, llm=llm)

        return cls(
            tools=tools,
            llm=llm,
            memory=memory,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
        )

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Chat history."""
        return self._memory.get_all()

    def reset(self) -> None:
        self._memory.reset()

    def _extract_reasoning_step(
        self, output: ChatResponse
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
            reasoning_step = self._output_parser.parse(message_content)
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

    def _process_actions(
        self, output: ChatResponse
    ) -> Tuple[List[BaseReasoningStep], bool]:
        _, current_reasoning, is_done = self._extract_reasoning_step(output)

        if is_done:
            return current_reasoning, True

        # call tool with input
        reasoning_step = cast(ActionReasoningStep, current_reasoning[-1])
        tool = self._tools_dict[reasoning_step.action]
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: reasoning_step.action_input,
                EventPayload.TOOL: tool.metadata,
            },
        ) as event:
            tool_output = tool.call(**reasoning_step.action_input)
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})

        observation_step = ObservationReasoningStep(observation=str(tool_output))
        current_reasoning.append(observation_step)
        if self._verbose:
            print_text(f"{observation_step.get_content()}\n", color="blue")
        return current_reasoning, False

    async def _aprocess_actions(
        self, output: ChatResponse
    ) -> Tuple[List[BaseReasoningStep], bool]:
        _, current_reasoning, is_done = self._extract_reasoning_step(output)

        if is_done:
            return current_reasoning, True

        # call tool with input
        reasoning_step = cast(ActionReasoningStep, current_reasoning[-1])
        tool = self._tools_dict[reasoning_step.action]
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: reasoning_step.action_input,
                EventPayload.TOOL: tool.metadata,
            },
        ) as event:
            tool_output = await tool.acall(**reasoning_step.action_input)
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})
        observation_step = ObservationReasoningStep(observation=str(tool_output))
        current_reasoning.append(observation_step)
        if self._verbose:
            print_text(f"{observation_step.get_content()}\n", color="blue")
        return current_reasoning, False

    def _get_response(
        self,
        current_reasoning: List[BaseReasoningStep],
    ) -> AgentChatResponse:
        """Get response from reasoning steps."""
        if len(current_reasoning) == 0:
            raise ValueError("No reasoning steps were taken.")
        elif len(current_reasoning) == self._max_iterations:
            raise ValueError("Reached max iterations.")

        response_step = cast(ResponseReasoningStep, current_reasoning[-1])

        # TODO: add sources from reasoning steps
        return AgentChatResponse(response=response_step.response, sources=[])

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """Chat."""
        if chat_history is not None:
            self._memory.set(chat_history)

        self._memory.put(ChatMessage(content=message, role="user"))

        current_reasoning: List[BaseReasoningStep] = []
        # start loop
        for _ in range(self._max_iterations):
            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                chat_history=self._memory.get(), current_reasoning=current_reasoning
            )
            # send prompt
            chat_response = self._llm.chat(input_chat)
            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = self._process_actions(output=chat_response)
            current_reasoning.extend(reasoning_steps)
            if is_done:
                break

        response = self._get_response(current_reasoning)
        self._memory.put(
            ChatMessage(content=response.response, role=MessageRole.ASSISTANT)
        )
        return response

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        self._memory.put(ChatMessage(content=message, role="user"))

        current_reasoning: List[BaseReasoningStep] = []
        # start loop
        for _ in range(self._max_iterations):
            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                chat_history=self._memory.get(), current_reasoning=current_reasoning
            )
            # send prompt
            chat_response = await self._llm.achat(input_chat)
            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = await self._aprocess_actions(
                output=chat_response
            )
            current_reasoning.extend(reasoning_steps)
            if is_done:
                break

        response = self._get_response(current_reasoning)
        self._memory.put(
            ChatMessage(content=response.response, role=MessageRole.ASSISTANT)
        )
        return response

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        current_reasoning: List[BaseReasoningStep] = []
        # start loop
        for _ in range(self._max_iterations):
            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                chat_history=self._memory.get(), current_reasoning=current_reasoning
            )
            # send prompt
            chat_stream = self._llm.stream_chat(input_chat)

            # iterate over stream, break out if is final answer after the "Answer: "
            is_done = False
            full_response = ChatResponse(
                message=ChatMessage(content=None, role="assistant")
            )
            for r in chat_stream:
                if "Answer: " in (r.message.content or ""):
                    is_done = True
                    break
                full_response = r
            if is_done:
                break

            # given react prompt outputs, call tools or return response
            reasoning_steps, _ = self._process_actions(output=full_response)
            current_reasoning.extend(reasoning_steps)

        # Get the response in a separate thread so we can yield the response
        chat_stream_response = StreamingAgentChatResponse(chat_stream=chat_stream)
        thread = Thread(
            target=chat_stream_response.write_response_to_history,
            args=(self._memory,),
        )
        thread.start()
        return chat_stream_response

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        self._memory.put(ChatMessage(content=message, role="user"))

        current_reasoning: List[BaseReasoningStep] = []
        # start loop
        for _ in range(self._max_iterations):
            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                chat_history=self._memory.get(), current_reasoning=current_reasoning
            )
            # send prompt
            chat_stream = await self._llm.astream_chat(input_chat)

            # iterate over stream, break out if is final answer
            is_done = False
            full_response = ChatResponse(
                message=ChatMessage(content=None, role="assistant")
            )
            async for r in chat_stream:
                if "Answer: " in (r.message.content or ""):
                    is_done = True
                    break
                full_response = r
            if is_done:
                break

            # given react prompt outputs, call tools or return response
            reasoning_steps, _ = self._process_actions(output=full_response)
            current_reasoning.extend(reasoning_steps)

        # Get the response in a separate thread so we can yield the response
        chat_stream_response = StreamingAgentChatResponse(achat_stream=chat_stream)
        # create task to write chat response to history
        asyncio.create_task(
            chat_stream_response.awrite_response_to_history(self._memory)
        )
        # thread.start()
        return chat_stream_response
