import asyncio
from typing import Any, List, Optional, Sequence
from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool, ToolOutput
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.output_parsers.base import BaseOutputParser
from llama_index.core.callbacks import CallbackManager, trace_method
import llama_index.core.instrumentation as instrument
import uuid

from .output_parser import CritiqueOutputParser, CorrectOutputParser

dispatcher = instrument.get_dispatcher(__name__)


class CriticAgentWorker(BaseAgentWorker):
    """CRITIC Agent Worker."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: FunctionCallingLLM,
        critique_prompt_template: str,
        critique_few_shot_examples: List[str],
        correct_prompt_template: str,
        critique_tools: Optional[Sequence[BaseTool]] = None,
        critique_output_parser: Optional[BaseOutputParser] = None,
        correct_output_parser: Optional[BaseOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.callback_manager = callback_manager or llm.callback_manager
        self.tools = tools
        self.critique_prompt_template = critique_prompt_template
        self.critique_few_shot_examples = critique_few_shot_examples
        self.correct_prompt_template = correct_prompt_template
        self.critique_tools = critique_tools
        self.critique_output_parser = critique_output_parser or CritiqueOutputParser
        self.correct_output_parser = correct_output_parser or CorrectOutputParser
        self.verbose = verbose

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        sources: List[ToolOutput] = []
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

        # put current history in new memory
        messages = task.memory.get()
        for message in messages:
            new_memory.put(message)

        # initialize task state
        task_state = {
            "sources": sources,
            "new_memory": new_memory,
        }
        task.extra_state.update(task_state)

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            step_state={"prev_reasoning": ""},
        )

    @dispatcher.span
    async def _arun_step(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        if self.verbose:
            print(f"> Running step {step.step_id} for task {task.task_id}.\n")

        # run CRITIC reflection
        num_iters = 3
        critic_history = []
        for i in range(num_iters):
            correct_prompt = ""
            critic_history.put(
                ChatMessage(role=MessageRole.USER, content=correct_prompt)
            )
            response = self.llm.predict_and_call(
                self.tools, chat_history=critic_history
            )
            parsed_correct_response = await self.correct_output_parser.aparse(
                response.message.content
            )
            critic_history.put(
                ChatMessage(role=MessageRole.ASSISTANT, content=parsed_correct_response)
            )

            # run critique step
            critique_prompt = ""
            critic_history.put(
                ChatMessage(role=MessageRole.USER, content=critique_prompt)
            )
            response = self.llm.predict_and_call(
                self.critique_tools, chat_history=critic_history
            )
            parsed_critique_response = await self.critique_output_parser.aparse(
                response.message.content
            )
            critic_history.put(
                ChatMessage(
                    role=MessageRole.ASSISTANT, content=parsed_critique_response
                )
            )

        # after CRITIC reflection and correction prepare TaskStepOutput
        task.extra_state["new_memory"].put(
            ChatMessage(content=parsed_correct_response), role=MessageRole.ASSISTANT
        )
        new_steps = [
            TaskStep(
                task_id=task.task_id,
                step_id=str(uuid.uuid4()),
                input=task.input,
                step_state={
                    "prev_reasoning": parsed_correct_response,
                },
            )
        ]
        return TaskStepOutput(
            output=parsed_correct_response, task_step=step, next_steps=new_steps
        )

    @dispatcher.span
    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        return asyncio.run(self.arun_step(step=step, task=task, **kwargs))

    @dispatcher.span
    @trace_method("run_step")
    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        return await self._arun_step(step, task)
