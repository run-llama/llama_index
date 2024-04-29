from llama_index.core.agent import (
    CustomSimpleAgentWorker,
    Task,
    AgentChatResponse,
)
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import BaseTool
from llama_index.core.llms import LLM
from llama_index.core.callbacks import (
    CallbackManager,
)
from llama_index.core.objects.base import ObjectRetriever
from typing import Any, Dict, Tuple, Sequence, Optional
from llama_index.core import Settings
from llama_index.core.agent.function_calling.step import FunctionCallingAgentWorker
from llama_index.core.bridge.pydantic import PrivateAttr, Field


from pydantic import BaseModel, Field


class Correction(BaseModel):
    """Data class for holding the corrected input."""

    correction: str = Field(default_factory=str, description="Corrected input")


class CriticAgentWorker(CustomSimpleAgentWorker):
    """Agent worker that combines tool calling with self-reflection.

    Continues iterating until there's no errors / task is done.

    """

    _max_iterations: int = PrivateAttr(default=5)
    _toxicity_threshold: float = PrivateAttr(default=3.0)
    _critique_agent_worker: FunctionCallingAgentWorker = PrivateAttr()
    _critique_template: str = PrivateAttr()

    def __init__(
        self,
        critique_agent_worker: FunctionCallingAgentWorker,
        critique_template: str,
        tools: Sequence[BaseTool],
        llm: LLM,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        **kwargs: Any,
    ) -> None:
        self._critique_agent_worker = critique_agent_worker
        self._critique_template = critique_template
        super().__init__(
            tools=tools,
            llm=llm,
            callback_manager=callback_manager or CallbackManager([]),
            tool_retriever=tool_retriever,
            verbose=verbose,
            **kwargs,
        )

    @classmethod
    def from_args(
        cls,
        critique_agent_worker: FunctionCallingAgentWorker,
        critique_template: str,
        tools: Optional[Sequence[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "CustomSimpleAgentWorker":
        """Convenience constructor method from set of of BaseTools (Optional)."""
        llm = llm or Settings.llm
        if callback_manager is not None:
            llm.callback_manager = callback_manager
        return cls(
            critique_agent_worker=critique_agent_worker,
            critique_template=critique_template,
            tools=tools or [],
            tool_retriever=tool_retriever,
            llm=llm,
            callback_manager=callback_manager or CallbackManager([]),
            verbose=verbose,
            **kwargs,
        )

    def _critique(self, input_str: str) -> AgentChatResponse:
        agent = self._critique_agent_worker.as_agent(verbose=True)
        critique = agent.chat(self._critique_template.format(input_str=input_str))
        print(f"Critique: {critique.response}", flush=True)
        return critique

    def _correct(self, input_str: str, critique: str) -> ChatMessage:
        from llama_index.llms.openai import OpenAI
        from llama_index.program.openai import OpenAIPydanticProgram

        correct_prompt_tmpl = """
        You are responsible for correcting an input based on a provided critique.

        Input:

        {input_str}

        Critique:

        {critique}

        Use the provided information to generate a corrected version of input.
        """

        correct_response_tmpl = (
            "Here is a corrected version of the input.\n{correction}"
        )

        correction_llm = OpenAI(model="gpt-4-turbo-preview", temperature=0)
        program = OpenAIPydanticProgram.from_defaults(
            Correction, prompt_template_str=correct_prompt_tmpl, llm=correction_llm
        )
        correction = program(input_str=input_str, critique=critique)
        print(f"Correction: {correction.correction}", flush=True)

        correct_response_str = correct_response_tmpl.format(
            correction=correction.correction
        )
        return ChatMessage.from_str(correct_response_str, role="assistant")

    def _initialize_state(self, task: Task, **kwargs: Any) -> Dict[str, Any]:
        """Initialize state."""
        return {"count": 0, "chat_history": []}

    def _run_step(
        self, state: Dict[str, Any], task: Task, input: Optional[str] = None
    ) -> Tuple[AgentChatResponse, bool]:
        """Run step."""
        # if first step, add assistant input
        if len(state["chat_history"]) == 0:
            state["chat_history"].append(
                ChatMessage.from_str(task.input, role="assistant")
            )

        current_response = state["chat_history"][-1].content
        # if reached max iters
        if state["count"] >= self._max_iterations:
            return AgentChatResponse(response=current_response), True

        # critique
        input_str = current_response.replace(
            "Here is a corrected version of the input.\n", ""
        )
        critique_response = self._critique(input_str=input_str)
        print(f"CRITIQUE SOURCES: {critique_response.sources}", flush=True)

        _, toxicity_score = critique_response.sources[0].raw_output
        print(f"toxicity_score: {toxicity_score}", flush=True)
        is_done = toxicity_score < self._toxicity_threshold

        critique_msg = ChatMessage(
            role=MessageRole.USER, content=critique_response.response
        )
        state["chat_history"].append(critique_msg)

        # correct
        if is_done:
            return AgentChatResponse(response=current_response), is_done
        else:
            correct_msg = self._correct(
                input_str=input_str, critique=critique_response.response
            )
            state["chat_history"].append(correct_msg)
            state["count"] += 1

        # return response
        return (
            AgentChatResponse(
                response=str(correct_msg), sources=critique_response.sources
            ),
            is_done,
        )

    def _finalize_task(self, state: Dict[str, Any], **kwargs) -> None:
        """Finalize task."""
