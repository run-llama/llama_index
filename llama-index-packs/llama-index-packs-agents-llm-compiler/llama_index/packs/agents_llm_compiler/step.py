"""
LLM Compiler.

A lot of this code was adapted from the source code of the LLM Compiler repo:
https://github.com/SqueezeAILab/LLMCompiler

"""

import asyncio
import uuid
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    cast,
)

from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.callbacks import (
    CallbackManager,
    trace_method,
)
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.program.llm_program import LLMTextCompletionProgram
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.tools import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.core.tools.types import AsyncBaseTool
from llama_index.core.utils import print_text
from llama_index.llms.openai import OpenAI

from .output_parser import (
    LLMCompilerJoinerParser,
    LLMCompilerPlanParser,
)
from .prompts import OUTPUT_PROMPT, PLANNER_EXAMPLE_PROMPT
from .schema import JoinerOutput
from .task_fetching_unit import (
    LLMCompilerTask,
    TaskFetchingUnit,
)
from .utils import (
    format_contexts,
    generate_context_for_replanner,
)

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
        tool_desc = (
            f"Tool Name: {tool.metadata.name}\n"
            f"Tool Description: {tool.metadata.description}\n"
            f"Tool Args: {tool.metadata.fn_schema_str}\n"
        )
        prefix += f"{i + 1}. {tool_desc}\n"

    # join operation
    prefix += f"{i + 2}. {JOIN_DESCRIPTION}\n\n"

    # Guidelines
    prefix += (
        "Guidelines:\n"
        " - Each action described above contains the tool name, description, and input schema.\n"
        "    - You must strictly adhere to the input types for each action.\n"
        "    - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.\n"
        "    - Do NOT specify arguments in kwargs format. Use positional arguments only.\n"
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
        prefix += "Here are some examples from other questions/toolsets.\n"
        prefix += f"Example:\n{example_prompt}\n\n"

    return prefix


class LLMCompilerAgentWorker(BaseAgentWorker):
    """
    LLMCompiler Agent Worker.

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
        planner_example_prompt_str: Optional[str] = None,
        stop: Optional[List[str]] = None,
        joiner_prompt: Optional[PromptTemplate] = None,
        max_replans: int = 3,
    ) -> None:
        self.callback_manager = callback_manager or llm.callback_manager

        self.planner_example_prompt_str = (
            planner_example_prompt_str or PLANNER_EXAMPLE_PROMPT
        )
        self.system_prompt = generate_llm_compiler_prompt(
            tools, example_prompt=self.planner_example_prompt_str
        )
        self.system_prompt_replan = generate_llm_compiler_prompt(
            tools, is_replan=True, example_prompt=self.planner_example_prompt_str
        )

        self.llm = llm
        # TODO: make tool_retriever work
        self.tools = tools
        self.output_parser = LLMCompilerPlanParser(tools=tools)
        self.stop = stop
        self.max_replans = max_replans
        self.verbose = verbose

        # joiner program
        self.joiner_prompt = joiner_prompt or PromptTemplate(OUTPUT_PROMPT)
        self.joiner_program = LLMTextCompletionProgram.from_defaults(
            output_parser=LLMCompilerJoinerParser(),
            output_cls=JoinerOutput,
            prompt=self.joiner_prompt,
            llm=self.llm,
            verbose=verbose,
        )

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
        """
        Convenience constructor method from set of BaseTools (Optional).

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
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

        # put user message in memory
        new_memory.put(ChatMessage(content=task.input, role=MessageRole.USER))

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
            step_state={"is_replan": False, "contexts": [], "replans": 0},
        )

    def get_tools(self, input: str) -> List[AsyncBaseTool]:
        """Get tools."""
        # return [adapt_to_async_tool(t) for t in self._get_tools(input)]
        return [adapt_to_async_tool(t) for t in self.tools]

    async def arun_llm(
        self,
        input: str,
        previous_context: Optional[str] = None,
        is_replan: bool = False,
    ) -> ChatResponse:
        """Run LLM."""
        if is_replan:
            system_prompt = self.system_prompt_replan
            assert previous_context is not None, "previous_context cannot be None"
            human_prompt = f"Question: {input}\n{previous_context}\n"
        else:
            system_prompt = self.system_prompt
            human_prompt = f"Question: {input}"

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=human_prompt),
        ]

        return await self.llm.achat(messages)

    async def ajoin(
        self,
        input: str,
        tasks: Dict[int, LLMCompilerTask],
        is_final: bool = False,
    ) -> JoinerOutput:
        """Join answer using LLM/agent."""
        agent_scratchpad = "\n\n"
        agent_scratchpad += "".join(
            [
                task.get_thought_action_observation(
                    include_action=True, include_thought=True
                )
                for task in tasks.values()
                if not task.is_join
            ]
        )
        agent_scratchpad = agent_scratchpad.strip()

        output = self.joiner_program(
            query_str=input,
            context_str=agent_scratchpad,
        )
        output = cast(JoinerOutput, output)
        if self.verbose:
            print_text(f"> Thought: {output.thought}\n", color="pink")
            print_text(f"> Answer: {output.answer}\n", color="pink")
        if is_final:
            output.is_replan = False
        return output

    def _get_task_step_response(
        self,
        task: Task,
        llmc_tasks: Dict[int, LLMCompilerTask],
        answer: str,
        joiner_thought: str,
        step: TaskStep,
        is_replan: bool,
    ) -> TaskStepOutput:
        """Get task step response."""
        agent_answer = AgentChatResponse(response=answer, sources=[])

        if not is_replan:
            # generate final answer
            new_steps = []

            # put in memory
            task.extra_state["new_memory"].put(
                ChatMessage(content=answer, role=MessageRole.ASSISTANT)
            )
        else:
            # Collect contexts for the subsequent replanner
            context = generate_context_for_replanner(
                tasks=llmc_tasks, joiner_thought=joiner_thought
            )
            new_contexts = step.step_state["contexts"] + [context]
            # TODO: generate new steps
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    input=None,
                    step_state={
                        "is_replan": is_replan,
                        "contexts": new_contexts,
                        "replans": step.step_state["replans"] + 1,
                    },
                )
            ]

        return TaskStepOutput(
            output=agent_answer,
            task_step=step,
            next_steps=new_steps,
            is_last=not is_replan,
        )

    async def _arun_step(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        if self.verbose:
            print(
                f"> Running step {step.step_id} for task {task.task_id}.\n"
                f"> Step count: {step.step_state['replans']}"
            )
        is_final_iter = (
            step.step_state["is_replan"]
            and step.step_state["replans"] >= self.max_replans
        )

        if len(step.step_state["contexts"]) == 0:
            formatted_contexts = None
        else:
            formatted_contexts = format_contexts(step.step_state["contexts"])
        llm_response = await self.arun_llm(
            task.input,
            previous_context=formatted_contexts,
            is_replan=step.step_state["is_replan"],
        )
        if self.verbose:
            print_text(f"> Plan: {llm_response.message.content}\n", color="pink")

        # return task dict (will generate plan, parse into dictionary)
        task_dict = self.output_parser.parse(cast(str, llm_response.message.content))

        # execute via task executor
        task_fetching_unit = TaskFetchingUnit.from_tasks(
            task_dict, verbose=self.verbose
        )
        await task_fetching_unit.schedule()

        ## join tasks - get response
        tasks = cast(Dict[int, LLMCompilerTask], task_fetching_unit.tasks)
        joiner_output = await self.ajoin(
            task.input,
            tasks,
            is_final=is_final_iter,
        )

        # get task step response (with new steps planned)
        return self._get_task_step_response(
            task,
            llmc_tasks=tasks,
            answer=joiner_output.answer,
            joiner_thought=joiner_output.thought,
            step=step,
            is_replan=joiner_output.is_replan,
        )

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        return asyncio.run(self.arun_step(step=step, task=task, **kwargs))

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
        task.memory.put_messages(task.extra_state["new_memory"].get_all())
        # reset new memory
        task.extra_state["new_memory"].reset()
