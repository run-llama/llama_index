"""Introspective agent worker."""

import logging
import uuid
from typing import Any, List, Optional, cast
import asyncio

from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.agent.utils import add_user_step_to_memory
from llama_index.core.callbacks import (
    CallbackManager,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool, adapt_to_async_tool
from llama_index.core.tools import BaseTool, adapt_to_async_tool
from llama_index.core.tools.types import AsyncBaseTool

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

DEFAULT_MAX_FUNCTION_CALLS = 5


def get_function_by_name(tools: List[BaseTool], name: str) -> BaseTool:
    """Get function by name."""
    name_to_tool = {tool.metadata.name: tool for tool in tools}
    if name not in name_to_tool:
        raise ValueError(f"Tool with name {name} not found")
    return name_to_tool[name]


class IntrospectiveAgentWorker(BaseAgentWorker):
    """Introspective Agent Worker.

    This agent worker implements the Reflectiong AI agentic pattern.
    """

    def __init__(
        self,
        tools: List[BaseTool],
        llm: FunctionCallingLLM,
        main_agent_worker: BaseAgentWorker,
        reflective_agent_worker: BaseAgentWorker,
        prefix_messages: List[ChatMessage],
        verbose: bool = False,
        max_function_calls: int = 5,
        callback_manager: Optional[CallbackManager] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        allow_parallel_tool_calls: bool = True,
    ) -> None:
        """Init params."""
        if not llm.metadata.is_function_calling_model:
            raise ValueError(
                f"Model name {llm.model} does not support function calling API. "
            )
        self._llm = llm
        self._verbose = verbose
        self._max_function_calls = max_function_calls
        self._main_agent_worker = main_agent_worker
        self._reflective_agent_worker = reflective_agent_worker
        self.prefix_messages = prefix_messages
        self.callback_manager = callback_manager or self._llm.callback_manager
        self.allow_parallel_tool_calls = allow_parallel_tool_calls

        if len(tools) > 0 and tool_retriever is not None:
            raise ValueError("Cannot specify both tools and tool_retriever")
        elif len(tools) > 0:
            self._get_tools = lambda _: tools
        elif tool_retriever is not None:
            tool_retriever_c = cast(ObjectRetriever[BaseTool], tool_retriever)
            self._get_tools = lambda message: tool_retriever_c.retrieve(message)
        else:
            # no tools
            self._get_tools = lambda _: []

    @classmethod
    def from_args(
        cls,
        main_agent_worker: BaseAgentWorker,
        reflective_agent_worker: BaseAgentWorker,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[FunctionCallingLLM] = None,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> "IntrospectiveAgentWorker":
        """Create an IntrospectiveAgentWorker from a list of tools.

        Similar to `from_defaults` in other classes, this method will
        infer defaults for a variety of parameters, including the LLM,
        if they are not specified.

        """
        tools = tools or []

        llm = llm or Settings.llm
        if callback_manager is not None:
            llm.callback_manager = callback_manager

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [ChatMessage(content=system_prompt, role="system")]

        prefix_messages = prefix_messages or []

        return cls(
            tools=tools,
            tool_retriever=tool_retriever,
            main_agent_worker=main_agent_worker,
            reflective_agent_worker=reflective_agent_worker,
            llm=llm,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
            **kwargs,
        )

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        # temporary memory for new messages
        main_memory = ChatMemoryBuffer.from_defaults()
        reflective_memory = ChatMemoryBuffer.from_defaults()

        # initialize task state
        task_state = {
            "main": {
                "memory": main_memory,
                "sources": [],
            },
            "reflection": {"memory": reflective_memory, "sources": []},
        }
        task.extra_state.update(task_state)

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
        )

    def get_tools(self, input: str) -> List[AsyncBaseTool]:
        """Get tools."""
        return [adapt_to_async_tool(t) for t in self._get_tools(input)]

    def get_all_messages(self, task: Task) -> List[ChatMessage]:
        return (
            self.prefix_messages
            + task.memory.get()
            + task.extra_state["main"]["memory"].get_all()
            + task.extra_state["reflection"]["memory"].get_all()
        )

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        # run main agent
        main_agent = self._main_agent_worker.as_agent()
        main_agent_response = main_agent.chat(
            task.input
        )  # or should i use step.input here?
        task.extra_state["main"]["sources"] = main_agent_response.sources
        task.extra_state["main"]["memory"] = main_agent.memory
        print(f"MAIN AGENT MEMORY: {main_agent.memory}", flush=True)

        # run reflective agent
        reflective_agent = self._reflective_agent_worker.as_agent()
        reflective_agent_response = reflective_agent.chat(main_agent_response.response)
        task.extra_state["reflection"]["sources"] = reflective_agent_response.sources
        task.extra_state["reflection"]["memory"] = reflective_agent.memory
        print(f"REFLECTIVE AGENT MEMORY: {reflective_agent.memory}", flush=True)

        agent_response = AgentChatResponse(
            response=str(reflective_agent_response.response),
            sources=task.extra_state["main"]["sources"]
            + task.extra_state["reflection"]["sources"],
        )

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=True,
            next_steps=[],
        )

    @trace_method("run_step")
    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        if step.input is not None:
            add_user_step_to_memory(
                step, task.extra_state["new_memory"], verbose=self._verbose
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)

        # get response and tool call (if exists)
        response = await self._llm.achat_with_tools(
            tools=tools,
            user_msg=None,
            chat_history=self.get_all_messages(task),
            verbose=self._verbose,
            allow_parallel_tool_calls=self.allow_parallel_tool_calls,
        )
        tool_calls = self._llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if self._verbose and response.message.content:
            print("=== LLM Response ===")
            print(str(response.message.content))

        if not self.allow_parallel_tool_calls and len(tool_calls) > 1:
            raise ValueError(
                "Parallel tool calls not supported for synchronous function calling agent"
            )

        # call all tools, gather responses
        task.extra_state["new_memory"].put(response.message)
        if (
            len(tool_calls) == 0
            or task.extra_state["n_function_calls"] >= self._max_function_calls
        ):
            # we are done
            is_done = True
            new_steps = []
        else:
            is_done = False
            tasks = [
                self._acall_function(
                    tools,
                    tool_call,
                    task.extra_state["new_memory"],
                    task.extra_state["sources"],
                    verbose=self._verbose,
                )
                for tool_call in tool_calls
            ]
            return_directs = await asyncio.gather(*tasks)

            # check if any of the tools return directly -- only works if there is one tool call
            if len(return_directs) == 1 and return_directs[0]:
                is_done = True
                response = task.extra_state["sources"][-1].content

            task.extra_state["n_function_calls"] += len(tool_calls)
            # put tool output in sources and memory
            new_steps = (
                [
                    step.get_next_step(
                        step_id=str(uuid.uuid4()),
                        # NOTE: input is unused
                        input=None,
                    )
                ]
                if not is_done
                else []
            )

        agent_response = AgentChatResponse(
            response=str(response), sources=task.extra_state["sources"]
        )

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    @trace_method("run_step")
    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        raise NotImplementedError("Stream not supported for function calling agent")

    @trace_method("run_step")
    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        raise NotImplementedError("Stream not supported for function calling agent")

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""
        # add new messages to memory
        task.memory.set(
            task.memory.get_all()
            + task.extra_state["main"]["memory"].get_all()
            + task.extra_state["reflection"]["memory"].get_all()
        )
        # reset new memory
        task.extra_state["main"]["memory"].reset()
