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
from llama_index.llms.base import ChatMessage, ChatResponse
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.indices.query.schema import QueryBundle
from llama_index.prompts.prompts import Prompt
from llama_index.response.schema import Response
from llama_index.agent.react.prompts import DEFAULT_REACT_PROMPT
from llama_index.agent.react.types import (
    BaseReasoningStep,
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from pydantic import BaseModel
from llama_index.callbacks.base import CallbackManager

from llama_index.agent.react.output_parser import ReActOutputParser


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
        react_prompt: Optional[Prompt] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._tools_dict = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history
        self._max_iterations = max_iterations
        self._react_prompt = react_prompt or DEFAULT_REACT_PROMPT
        self._output_parser = output_parser or ReActOutputParser()
        self.callback_manager = callback_manager or CallbackManager([])

    def chat_history(self) -> CHAT_HISTORY_TYPE:
        """Chat history."""
        return self._chat_history

    def _get_inputs(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ) -> str:
        """Get inputs."""
        current_reasoning = current_reasoning or []
        current_reasoning_str = "\n".join(r for r in current_reasoning)
        chat_history_str = "\n".join(
            [f"> {chat_message.content}" for chat_message in chat_history]
        )
        return self._react_prompt.format(
            tool_desc="\n".join(
                [
                    f"> {tool.metadata.name}: {tool.metadata.description}"
                    for tool in self._tools
                ]
            ),
            tool_names=", ".join([tool.metadata.name for tool in self._tools]),
            chat_history_str=chat_history_str,
            new_message=message,
            current_reasoning=current_reasoning_str,
        )

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
            input_prompt = self._get_inputs(
                message, chat_history, current_reasoning=current_reasoning
            )
            # send prompt
            chat_response = self._llm.chat(input_prompt)
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
