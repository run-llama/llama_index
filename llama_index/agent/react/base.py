# ReAct agent

from llama_index.agent.openai_agent import OpenAIAgent
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms.base import LLM
from typing import Sequence, cast
from llama_index.tools import BaseTool
from llama_index.agent.types import BaseAgent, CHAT_HISTORY_TYPE
from abc import abstractmethod
from typing import List, Optional
from llama_index.llms.base import ChatMessage, ChatResponse, MessageRole
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.indices.query.schema import QueryBundle
from llama_index.prompts.prompts import Prompt
from llama_index.response.schema import Response
from llama_index.agent.react.prompts import (
    # DEFAULT_REACT_PROMPT,
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

# from langchain.agents.conversational i

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
            "\n".join(r for r in current_reasoning) if current_reasoning else "None"
        )

        fmt_sys_header = self.system_header.format(
            tool_desc="\n".join(
                [
                    f"> {tool.metadata.name}: {tool.metadata.description}"
                    for tool in self.tools
                ]
            ),
            tool_names=", ".join([tool.metadata.name for tool in self.tools]),
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

    Works for all non-completion prompt types.

    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        chat_history: List[ChatMessage],
        max_iterations: Optional[int] = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
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

    def reset(self) -> None:
        self._chat_history.clear()

    @classmethod
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        max_iterations: Optional[int] = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
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
        )

    def chat_history(self) -> CHAT_HISTORY_TYPE:
        """Chat history."""
        return self._chat_history

    # def _get_inputs(
    #     self,
    #     message: str,
    #     chat_history: Optional[List[ChatMessage]],
    #     current_reasoning: Optional[List[BaseReasoningStep]] = None,
    # ) -> str:
    #     """Get inputs."""
    #     current_reasoning = current_reasoning or []
    #     current_reasoning_str = "\n".join(r for r in current_reasoning)
    #     chat_history_str = "\n".join(
    #         [f"> {chat_message.content}" for chat_message in chat_history]
    #     )
    #     return self._react_prompt.format(
    #         tool_desc="\n".join(
    #             [
    #                 f"> {tool.metadata.name}: {tool.metadata.description}"
    #                 for tool in self._tools
    #             ]
    #         ),
    #         tool_names=", ".join([tool.metadata.name for tool in self._tools]),
    #         chat_history_str=chat_history_str,
    #         new_message=message,
    #         current_reasoning=current_reasoning_str,
    #     )

    def _parse_output(self, output_message: ChatMessage) -> BaseReasoningStep:
        """Parse output."""
        content = output_message.content
        # check if observation
        print(content)
        raise Exception

    def _process_actions(
        self, output: ChatResponse, current_reasoning: List[BaseReasoningStep]
    ) -> List[BaseReasoningStep]:
        """Process outputs (and execute tools)."""
        ai_message = output.message
        # parse output
        reasoning_step = self._parse_output(ai_message)
        current_reasoning.append(reasoning_step)
        if reasoning_step.is_done:
            return current_reasoning

        reasoning_step = cast(ActionReasoningStep, reasoning_step)
        if not isinstance(reasoning_step, ActionReasoningStep):
            raise ValueError(f"Expected ActionReasoningStep, got {reasoning_step}")
        # call tool with input
        tool = self._tools_dict[reasoning_step.action]

        output = tool(**reasoning_step.action_input)

        current_reasoning.append(output.message.content)
        return current_reasoning

    def _get_response(
        self,
        current_reasoning: List[BaseReasoningStep],
        chat_history: List[ChatMessage],
    ) -> List[BaseReasoningStep]:
        """Get response from reasoning steps."""
        if len(current_reasoning) == 0:
            raise ValueError("No reasoning steps were taken.")
        elif len(current_reasoning) == self._max_iterations:
            raise ValueError("Reached max iterations.")

        response_text = current_reasoning[-1].get_content()

        return Response(response=response_text)

    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        """Chat."""
        chat_history = chat_history or self._chat_history
        chat_history.append(ChatMessage(content=message, role="user"))

        current_reasoning = []
        # start loop
        for _ in range(self._max_iterations):
            # prepare inputs
            input_chat = self._react_chat_formatter.format(
                chat_history=chat_history, current_reasoning=current_reasoning
            )
            # input_prompt = self._get_inputs(
            #     message, chat_history, current_reasoning=current_reasoning
            # )
            # send prompt
            chat_response = self._llm.chat(input_chat)
            reasoning_steps = self._process_actions(
                output=chat_response, current_reasoning=current_reasoning
            )
            current_reasoning.extend(reasoning_steps)

        return self._get_response(current_reasoning, chat_history)

    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        chat_history = chat_history or self._chat_history
        chat_history.append(ChatMessage(content=message, role="user"))

        current_reasoning = []
        # start loop
        for _ in range(self._max_iterations):
            # prepare inputs
            input_prompt = self._get_inputs(
                message, chat_history, current_reasoning=current_reasoning
            )
            # send prompt
            chat_response = await self._llm.achat(input_prompt)
            reasoning_steps = self._process_actions(
                output=chat_response, current_reasoning=current_reasoning
            )
            current_reasoning.extend(reasoning_steps)

        return self._get_response(current_reasoning)

    # ===== Query Engine Interface =====
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self.chat(
            query_bundle.query_str,
            chat_history=[],
        )

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return await self.achat(
            query_bundle.query_str,
            chat_history=[],
        )
