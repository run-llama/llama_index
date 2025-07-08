"""Function calling agent worker."""

import json
import logging
import uuid
from typing import Any, List, Optional, Sequence, cast
import asyncio
import llama_index.core.instrumentation as instrument
from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.agent.utils import add_user_step_to_memory
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.llms.function_calling import FunctionCallingLLM, ToolSelection
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.core.tools.calling import (
    call_tool_with_selection,
    acall_tool_with_selection,
)
from llama_index.core.tools import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.core.tools.types import AsyncBaseTool, ToolMetadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

dispatcher = instrument.get_dispatcher(__name__)

DEFAULT_MAX_FUNCTION_CALLS = 5


def get_function_by_name(tools: Sequence[BaseTool], name: str) -> Optional[BaseTool]:
    """Get function by name. If the function is not found, None is returned."""
    name_to_tool = {tool.metadata.name: tool for tool in tools}
    return name_to_tool.get(name)


def build_missing_tool_message(missing_tool_name: str) -> str:
    """
    Build an error message for the case where a tool is not found. This message
    instructs the LLM to double check the tool name, since it was hallucinated.
    """
    return f"Tool with name {missing_tool_name} not found, please double check."


def build_error_tool_output(tool_name: str, tool_args: Any, err_msg: str) -> ToolOutput:
    """Build a ToolOutput for an error that has occurred."""
    return ToolOutput(
        content=err_msg,
        tool_name=tool_name,
        raw_input={"args": str(tool_args)},
        raw_output=err_msg,
        is_error=True,
    )


def build_missing_tool_output(bad_tool_call: ToolSelection) -> ToolOutput:
    """
    Build a ToolOutput for the case where a tool is not found. This output contains
    instructions that ask the LLM to double check the tool name, along with the
    hallucinated tool name itself.
    """
    return build_error_tool_output(
        bad_tool_call.tool_name,
        bad_tool_call.tool_kwargs,
        build_missing_tool_message(bad_tool_call.tool_name),
    )


class FunctionCallingAgentWorker(BaseAgentWorker):
    """Function calling agent worker."""

    def __init__(
        self,
        tools: List[BaseTool],
        llm: FunctionCallingLLM,
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
                f"Model name {llm.metadata.model_name} does not support function calling API. "
            )
        self._llm = llm
        self._verbose = verbose
        self._max_function_calls = max_function_calls
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
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[FunctionCallingLLM] = None,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        allow_parallel_tool_calls: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> "FunctionCallingAgentWorker":
        """
        Create an FunctionCallingAgentWorker from a list of tools.

        Similar to `from_defaults` in other classes, this method will
        infer defaults for a variety of parameters, including the LLM,
        if they are not specified.

        """
        tools = tools or []

        llm = llm or Settings.llm  # type: ignore
        assert isinstance(llm, FunctionCallingLLM), (
            "llm must be an instance of FunctionCallingLLM"
        )

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
            llm=llm,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        sources: List[ToolOutput] = []
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()
        # initialize task state
        task_state = {
            "sources": sources,
            "n_function_calls": 0,
            "new_memory": new_memory,
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
            + task.memory.get(input=task.input)
            + task.extra_state["new_memory"].get_all()
        )

    def _call_function(
        self,
        tools: Sequence[BaseTool],
        tool_call: ToolSelection,
        memory: BaseMemory,
        sources: List[ToolOutput],
        verbose: bool = False,
    ) -> bool:
        tool = get_function_by_name(tools, tool_call.tool_name)
        tool_args_str = json.dumps(tool_call.tool_kwargs)
        tool_metadata = (
            tool.metadata
            if tool is not None
            else ToolMetadata(description="", name=tool_call.tool_name)
        )

        dispatcher.event(
            AgentToolCallEvent(arguments=tool_args_str, tool=tool_metadata)
        )
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: tool_args_str,
                EventPayload.TOOL: tool_metadata,
            },
        ) as event:
            tool_output = (
                call_tool_with_selection(tool_call, tools, verbose=verbose)
                if tool is not None
                else build_missing_tool_output(tool_call)
            )
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})

        function_message = ChatMessage(
            content=str(tool_output),
            role=MessageRole.TOOL,
            additional_kwargs={
                "name": tool_call.tool_name,
                "tool_call_id": tool_call.tool_id,
            },
        )
        sources.append(tool_output)
        memory.put(function_message)

        return tool.metadata.return_direct if tool is not None else False

    async def _acall_function(
        self,
        tools: Sequence[BaseTool],
        tool_call: ToolSelection,
        memory: BaseMemory,
        sources: List[ToolOutput],
        verbose: bool = False,
    ) -> bool:
        tool = get_function_by_name(tools, tool_call.tool_name)
        tool_args_str = json.dumps(tool_call.tool_kwargs)
        tool_metadata = (
            tool.metadata
            if tool is not None
            else ToolMetadata(description="", name=tool_call.tool_name)
        )

        dispatcher.event(
            AgentToolCallEvent(arguments=tool_args_str, tool=tool_metadata)
        )
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: tool_args_str,
                EventPayload.TOOL: tool_metadata,
            },
        ) as event:
            tool_output = (
                await acall_tool_with_selection(tool_call, tools, verbose=verbose)
                if tool is not None
                else build_missing_tool_output(tool_call)
            )
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})

        function_message = ChatMessage(
            content=str(tool_output),
            role=MessageRole.TOOL,
            additional_kwargs={
                "name": tool_call.tool_name,
                "tool_call_id": tool_call.tool_id,
            },
        )
        sources.append(tool_output)
        memory.put(function_message)

        return tool.metadata.return_direct if tool is not None else False

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_memory(
                step, task.extra_state["new_memory"], verbose=self._verbose
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)

        # get response and tool call (if exists)
        response = self._llm.chat_with_tools(
            tools=tools,
            user_msg=None,
            chat_history=self.get_all_messages(task),
            verbose=self._verbose,
            allow_parallel_tool_calls=self.allow_parallel_tool_calls,
        )
        tool_calls = self._llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        tool_outputs: List[ToolOutput] = []

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
            for i, tool_call in enumerate(tool_calls):
                # TODO: maybe execute this with multi-threading
                return_direct = self._call_function(
                    tools,
                    tool_call,
                    task.extra_state["new_memory"],
                    tool_outputs,
                    verbose=self._verbose,
                )
                task.extra_state["sources"].append(tool_outputs[-1])
                task.extra_state["n_function_calls"] += 1

                # check if any of the tools return directly -- only works if there is one tool call
                if i == 0 and return_direct:
                    is_done = True
                    response = task.extra_state["sources"][-1].content
                    break

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

        # get response string
        # return_direct can change the response type
        try:
            response_str = str(response.message.content)
        except AttributeError:
            response_str = str(response)

        agent_response = AgentChatResponse(response=response_str, sources=tool_outputs)

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
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
        tool_outputs: List[ToolOutput] = []

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
                    tool_outputs,
                    verbose=self._verbose,
                )
                for tool_call in tool_calls
            ]
            return_directs = await asyncio.gather(*tasks)
            task.extra_state["sources"].extend(tool_outputs)

            # check if any of the tools return directly -- only works if there is one tool call
            if len(return_directs) == 1 and return_directs[0]:
                is_done = True
                response = tool_outputs[-1].content  # type: ignore

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

        # get response string
        # return_direct can change the response type
        try:
            response_str = str(response.message.content)
        except AttributeError:
            response_str = str(response)

        agent_response = AgentChatResponse(response=response_str, sources=tool_outputs)

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
            task.memory.get_all() + task.extra_state["new_memory"].get_all()
        )
        # reset new memory
        task.extra_state["new_memory"].reset()
