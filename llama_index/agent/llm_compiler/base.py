"""LLM Compiler."""

import asyncio
import uuid
from itertools import chain
from threading import Thread
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from llama_index.agent.react.formatter import ReActChatFormatter
from llama_index.agent.react.output_parser import ReActOutputParser
from llama_index.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.llms.base import ChatMessage, ChatResponse
from llama_index.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.types import MessageRole
from llama_index.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.memory.types import BaseMemory
from llama_index.objects.base import ObjectRetriever
from llama_index.tools import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.tools.types import AsyncBaseTool
from llama_index.utils import print_text, unit_generator
from llama_index.prompts.base import PromptTemplate

DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"

JOIN_DESCRIPTION = (
    "join():\n"
    " - Collects and combines results from prior actions.\n"
    " - A LLM agent is called upon invoking join to either finalize the user query or wait until the plans are executed.\n"
    " - join should always be the last action in the plan, and will be called in two scenarios:\n"
    "   (a) if the answer can be determined by gathering the outputs from tasks to generate the final response.\n"
    "   (b) if the answer cannot be determined in the planning phase before you execute the plans. "
)

END_OF_PLAN = "<END_OF_PLAN>"

def generate_llm_compiler_prompt(
    tools: Sequence[BaseTool],
    is_replan: bool = False,
    example_prompt: Optional[str] = None,
) -> str:
    """Generate LLM Compiler prompt."""
    prefix = (
        "Given a user query, create a plan to solve it with the utmost parallelizability. "
        f"Each plan should comprise an action from the following {len(tools) + 1} types:\n"
    )

    # tools
    for i, tool in enumerate(tools):
        prefix += f"{i + 1}. {tool.metadata.description}\n"

    # join operation
    prefix += f"{i+2}. {JOIN_DESCRIPTION}\n\n"

    # Guidelines
    prefix += (
        "Guidelines:\n"
        " - Each action described above contains input/output types and description.\n"
        "    - You must strictly adhere to the input and output types for each action.\n"
        "    - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.\n"
        " - Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.\n"
        " - Each action MUST have a unique ID, which is strictly increasing.\n"
        " - Inputs for actions can either be constants or outputs from preceding actions. "
        "In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.\n"
        f" - Always call join as the last action in the plan. Say '{END_OF_PLAN}' after you call join\n"
        " - Ensure the plan maximizes parallelizability.\n"
        " - Only use the provided action types. If a query cannot be addressed using these, invoke the join action for the next steps.\n"
        " - Never explain the plan with comments (e.g. #).\n"
        " - Never introduce new actions other than the ones provided.\n\n"
    )

    if is_replan:
        prefix += (
            ' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
            "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
            'You MUST use these information to create the next plan under "Current Plan".\n'
            ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
            " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
        )

    if example_prompt is not None:
        prefix += f"Example:\n{example_prompt}\n\n"

    return prefix


class LLMCompilerAgentWorker(BaseAgentWorker):
    """LLMCompiler Agent Worker.

    LLMCompiler is an agent framework that allows async multi-function calling and query planning.
    Here is the implementation.

    Source Repo (paper linked): https://github.com/SqueezeAILab/LLMCompiler?tab=readme-ov-file

    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        stop: Optional[List[str]] = None,
    ) -> None:
        self.callback_manager = callback_manager or llm.callback_manager
        
        self.system_prompt = PromptTemplate(generate_llm_compiler_prompt(tools))
        self.system_prompt_replan = PromptTemplate(
            generate_llm_compiler_prompt(tools, is_replan=True)
        )

        # TODO: make tool_retriever work
        self.tools = tools

        self.output_parser = LLMCompilerPlanParser(tools=tools)
        self.stop = stop

        # if len(tools) > 0 and tool_retriever is not None:
        #     raise ValueError("Cannot specify both tools and tool_retriever")
        # elif len(tools) > 0:
        #     self._get_tools = lambda _: tools
        # elif tool_retriever is not None:
        #     tool_retriever_c = cast(ObjectRetriever[BaseTool], tool_retriever)
        #     self._get_tools = lambda message: tool_retriever_c.retrieve(message)
        # else:
        #     self._get_tools = lambda _: []

    @classmethod
    def from_tools(
        cls,
        tools: Optional[Sequence[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "LLMCompilerAgentWorker":
        """Convenience constructor method from set of of BaseTools (Optional).

        Returns:
            LLMCompilerAgentWorker: the LLMCompilerAgentWorker instance

        """
        llm = llm or OpenAI(model=DEFAULT_MODEL_NAME)
        if callback_manager is not None:
            llm.callback_manager = callback_manager
        return cls(
            tools=tools or [],
            tool_retriever=tool_retriever,
            llm=llm,
            callback_manager=callback_manager,
            verbose=verbose,
        )

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        sources: List[ToolOutput] = []
        current_reasoning: List[BaseReasoningStep] = []
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

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
            step_state={"is_first": True},
        )

    def get_tools(self, input: str) -> List[AsyncBaseTool]:
        """Get tools."""
        # return [adapt_to_async_tool(t) for t in self._get_tools(input)]
        return [adapt_to_async_tool(t) for t in self.tools]

    def _run_step(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        

    # async def _arun_step(
    #     self,
    #     step: TaskStep,
    #     task: Task,
    # ) -> TaskStepOutput:
    #     """Run step."""

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        return self._run_step(step, task)

    @trace_method("run_step")
    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        return await self._arun_step(step, task)

    @trace_method("run_step")
    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        # # TODO: figure out if we need a different type for TaskStepOutput
        # return self._run_step_stream(step, task)
        raise NotImplementedError

    @trace_method("run_step")
    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        raise NotImplementedError
        # """Run step (async stream)."""
        # return await self._arun_step_stream(step, task)

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""
        # add new messages to memory
        task.memory.set(task.memory.get() + task.extra_state["new_memory"].get_all())
        # reset new memory
        task.extra_state["new_memory"].reset()
