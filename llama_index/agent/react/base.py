# ReAct agent

from llama_index.llms.base import LLM
from typing import Sequence, cast, Tuple
from llama_index.tools import BaseTool
from llama_index.agent.types import BaseAgent
from abc import abstractmethod
from typing import List, Optional
from llama_index.llms.base import ChatMessage, ChatResponse, MessageRole
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.response.schema import Response
from llama_index.agent.react.prompts import (
    REACT_CHAT_SYSTEM_HEADER,
    REACT_CHAT_LAST_USER_MESSAGE,
)
from llama_index.agent.react.types import (
    BaseReasoningStep,
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from pydantic import BaseModel
from llama_index.callbacks.base import CallbackManager
from llama_index.llms.openai import OpenAI

from llama_index.agent.react.output_parser import ReActOutputParser
from llama_index.bridge.langchain import print_text


def get_react_tool_descriptions(tools: Sequence[BaseTool]) -> List[str]:
    """Tool"""
    tool_descs = []
    for tool in tools:
        tool_desc = (
            f"> Tool Name: {tool.metadata.name}\n"
            f"Tool Description: {tool.metadata.description}\n"
            f"Tool Args: {tool.metadata.fn_schema_str}\n"
        )
        tool_descs.append(tool_desc)
    return tool_descs


# TODO: come up with better name
# TODO: move into module
class BaseAgentChatFormatter(BaseModel):
    """Base chat formatter."""

    tools: Sequence[BaseTool]

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def format(
        self,
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ) -> List[ChatMessage]:
        """Format chat history into list of ChatMessage."""


class ReActChatFormatter(BaseAgentChatFormatter):
    """ReAct chat formatter."""

    system_header: str = REACT_CHAT_SYSTEM_HEADER
    last_user_message: str = REACT_CHAT_LAST_USER_MESSAGE

    def format(
        self,
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ) -> List[ChatMessage]:
        """Format chat history into list of ChatMessage."""
        current_reasoning = current_reasoning or []
        current_reasoning_str = (
            "\n".join(r.get_content() for r in current_reasoning)
            if current_reasoning
            else "None"
        )

        tool_descs_str = "\n".join(get_react_tool_descriptions(self.tools))

        fmt_sys_header = self.system_header.format(
            tool_desc=tool_descs_str,
            tool_names=", ".join([tool.metadata.get_name() for tool in self.tools]),
        )
        prev_chat_history = chat_history[:-1]
        last_chat = chat_history[-1]

        fmt_last_user_message = self.last_user_message.format(
            new_message=last_chat.content,
            current_reasoning=current_reasoning_str,
        )

        formatted_chat = [
            ChatMessage(role=MessageRole.SYSTEM, content=fmt_sys_header),
            *prev_chat_history,
            ChatMessage(role=MessageRole.USER, content=fmt_last_user_message),
        ]
        return formatted_chat


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
        chat_history: List[ChatMessage],
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._tools_dict = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history
        self._max_iterations = max_iterations
        self._react_chat_formatter = react_chat_formatter or ReActChatFormatter(
            tools=tools
        )
        self._output_parser = output_parser or ReActOutputParser()
        self.callback_manager = callback_manager or CallbackManager([])
        self._verbose = verbose

    def reset(self) -> None:
        self._chat_history.clear()

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
    ) -> "ReActAgent":
        tools = tools or []
        chat_history = chat_history or []
        llm = llm or OpenAI(model="gpt-3.5-turbo-0613")

        if not isinstance(llm, OpenAI):
            raise ValueError(f"Expected OpenAI, got {llm}")

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
        return self._chat_history

    def _process_actions(
        self, output: ChatResponse
    ) -> Tuple[List[BaseReasoningStep], bool]:
        """Process outputs (and execute tools)."""
        # TODO: remove Optional typing from message?
        message_content = cast(str, output.message.content)
        # parse output into either an ActionReasoningStep or ResponseReasoningStep
        current_reasoning = []
        reasoning_step = self._output_parser.parse(message_content)
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

        output = tool(**reasoning_step.action_input)
        observation_step = ObservationReasoningStep(observation=str(output))
        current_reasoning.append(observation_step)
        if self._verbose:
            print_text(f"{observation_step.get_content()}\n", color="blue")
        return current_reasoning, False

    def _get_response(
        self,
        current_reasoning: List[BaseReasoningStep],
    ) -> Response:
        """Get response from reasoning steps."""
        if len(current_reasoning) == 0:
            raise ValueError("No reasoning steps were taken.")
        elif len(current_reasoning) == self._max_iterations:
            raise ValueError("Reached max iterations.")

        response_step = cast(ResponseReasoningStep, current_reasoning[-1])

        return Response(response=response_step.response)

    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        """Chat."""
        chat_history = chat_history or self._chat_history
        chat_history.append(ChatMessage(content=message, role="user"))

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
        chat_history.append(
            ChatMessage(content=response.response, role=MessageRole.ASSISTANT)
        )
        return response

    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        chat_history = chat_history or self._chat_history
        chat_history.append(ChatMessage(content=message, role="user"))

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
        chat_history.append(
            ChatMessage(content=response.response, role=MessageRole.ASSISTANT)
        )
        return response
