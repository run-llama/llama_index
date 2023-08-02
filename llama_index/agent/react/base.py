# ReAct agent

import time
from threading import Thread
from typing import Any, List, Optional, Sequence, Tuple, cast

from llama_index.agent.react.formatter import ReActChatFormatter
from llama_index.agent.react.output_parser import ReActOutputParser
from llama_index.agent.react.types import (ActionReasoningStep,
                                           BaseReasoningStep,
                                           ObservationReasoningStep,
                                           ResponseReasoningStep)
from llama_index.agent.types import BaseAgent
from llama_index.bridge.langchain import print_text
from llama_index.callbacks.base import CallbackManager
from llama_index.chat_engine.types import (AgentChatResponse,
                                           StreamingAgentChatResponse)
from llama_index.llms.base import LLM, ChatMessage, ChatResponse, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.memory.types import BaseMemory
from llama_index.tools import BaseTool


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
        self._tools = tools
        self._tools_dict = {tool.metadata.name: tool for tool in tools}
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
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "ReActAgent":
        tools = tools or []
        chat_history = chat_history or []
        llm = llm or OpenAI(model="gpt-3.5-turbo-0613")

        return cls(
            tools=tools,
            llm=llm,
            chat_history=chat_history,
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

    def _process_actions(
        self, output: ChatResponse
    ) -> Tuple[List[BaseReasoningStep], bool]:
        """Process outputs (and execute tools)."""
        if output.message.content is None:
            raise ValueError("Got empty message.")
        message_content = output.message.content
        # parse output into either an ActionReasoningStep or ResponseReasoningStep
        current_reasoning = []
        try:
            reasoning_step = self._output_parser.parse(message_content)
        except BaseException:
            raise ValueError(f"Could not parse output: {message_content}")
        if self._verbose:
            print_text(f"{reasoning_step.get_content()}\n", color="pink")
        current_reasoning.append(reasoning_step)
        # is done if ResponseReasoningStep
        if reasoning_step.is_done:
            return current_reasoning, True

        reasoning_step = cast(ActionReasoningStep, reasoning_step)
        if not isinstance(reasoning_step, ActionReasoningStep):
            raise ValueError(f"Expected ActionReasoningStep, got {reasoning_step}")
        # call tool with input
        tool = self._tools_dict[reasoning_step.action]

        tool_output = tool(**reasoning_step.action_input)
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

    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """Chat."""
        if chat_history is not None:
            self._memory.set(chat_history)

        self._memory.set(ChatMessage(content=message, role="user"))

        current_reasoning: List[BaseReasoningStep] = []
        # start loop
        for _ in range(self._max_iterations):
            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                chat_history=chat_history, current_reasoning=current_reasoning
            )
            # send prompt
            chat_response = self._llm.chat(input_chat)
            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = self._process_actions(output=chat_response)
            current_reasoning.extend(reasoning_steps)
            if is_done:
                break

        response = self._get_response(current_reasoning)
        self._memory.set(
            ChatMessage(content=response.response, role=MessageRole.ASSISTANT)
        )
        return response

    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        self._memory.set(ChatMessage(content=message, role="user"))

        current_reasoning: List[BaseReasoningStep] = []
        # start loop
        for _ in range(self._max_iterations):
            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                chat_history=chat_history, current_reasoning=current_reasoning
            )
            # send prompt
            chat_response = await self._llm.achat(input_chat)
            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = self._process_actions(output=chat_response)
            current_reasoning.extend(reasoning_steps)
            if is_done:
                break

        response = self._get_response(current_reasoning)
        self._memory.set(
            ChatMessage(content=response.response, role=MessageRole.ASSISTANT)
        )
        return response

    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        # chat_history = chat_history or self._chat_history
        # chat_history.append(ChatMessage(content=message, role="user"))

        # current_reasoning: List[BaseReasoningStep] = []
        # # start loop
        # for _ in range(self._max_iterations):
        #     # prepare inputs
        #     input_chat = self._react_chat_formatter.format(
        #         chat_history=chat_history, current_reasoning=current_reasoning
        #     )
        #     # send prompt
        #     chat_response = self._llm.chat(input_chat)
        #     # given react prompt outputs, call tools or return response
        #     reasoning_steps, is_done = self._process_actions(output=chat_response)
        #     current_reasoning.extend(reasoning_steps)
        #     if is_done:
        #         break

        # response = self._get_response(current_reasoning)
        # chat_history.append(
        #     ChatMessage(content=response.response, role=MessageRole.ASSISTANT)
        # )
        ########

        if chat_history is not None:
            self._memory.set(chat_history)
        all_messages = self._memory.get()
        sources = []

        current_reasoning: List[BaseReasoningStep] = []
        # prepare inputs
        input_chat = self._react_chat_formatter.format(
            chat_history=self._memory.get(), current_reasoning=current_reasoning
        )
        chat_stream = self._llm.stream_chat(input_chat)
        chat_stream_response = StreamingAgentChatResponse(chat_stream=chat_stream)

        # Get the response in a separate thread so we can yield the response
        thread = Thread(
            target=chat_stream_response.write_response_to_history,
            args=(self._memory,),
        )
        thread.start()

        while chat_stream_response._is_final is None:
            # Wait until we know if the response is a function call or not
            time.sleep(0.05)
            if chat_stream_response._is_final is True:
                return chat_stream_response

        thread.join()

        n_function_calls = 0
        function_call_ = self._get_latest_function_call(self._memory.get_all())
        # start loop
        for _ in range(self._max_iterations):
            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                chat_history=chat_history, current_reasoning=current_reasoning
            )
            # send prompt
            chat_response = self._llm.chat(input_chat)
            # given react prompt outputs, call tools or return response
            reasoning_steps, is_done = self._process_actions(output=chat_response)
            current_reasoning.extend(reasoning_steps)
            if is_done:
                break

        while function_call_ is not None:
            if n_function_calls >= self._max_function_calls:
                print(f"Exceeded max function calls: {self._max_function_calls}.")
                break

            function_message, tool_output = call_function(
                tools, function_call_, verbose=self._verbose
            )
            sources.append(tool_output)
            self._memory.put(function_message)
            n_function_calls += 1

            # send function call & output back to get another response
            all_messages = self._prefix_messages + self._memory.get()
            chat_stream_response = StreamingAgentChatResponse(
                chat_stream=self._llm.stream_chat(all_messages, functions=functions),
                sources=sources,
            )

            # Get the response in a separate thread so we can yield the response
            thread = Thread(
                target=chat_stream_response.write_response_to_history,
                args=(self._memory,),
            )
            thread.start()
            while chat_stream_response._is_function is None:
                # Wait until we know if the response is a function call or not
                time.sleep(0.05)
                if chat_stream_response._is_function is False:
                    return chat_stream_response

            thread.join()
            function_call_ = self._get_latest_function_call(self._memory.get_all())

    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("astream_chat not implemented")
