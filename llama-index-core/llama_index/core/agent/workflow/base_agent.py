from abc import abstractmethod
import functools
import warnings
import inspect
from typing import Any, Callable, Dict, List, Sequence, Optional, Union, Type, cast

from pydantic._internal._model_construction import ModelMetaclass
from llama_index.core.agent.workflow.prompts import DEFAULT_STATE_PROMPT
from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput,
    AgentInput,
    AgentSetup,
    AgentWorkflowStartEvent,
    AgentStreamStructuredOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.agent.utils import generate_structured_response
from llama_index.core.llms import ChatMessage, LLM, TextBlock
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptMixin, PromptMixinType, PromptDictType
from llama_index.core.tools import (
    BaseTool,
    AsyncBaseTool,
    FunctionTool,
    ToolOutput,
    ToolSelection,
    adapt_to_async_tool,
)
from llama_index.core.workflow import Context
from llama_index.core.objects import ObjectRetriever
from llama_index.core.settings import Settings
from llama_index.core.workflow.context import Context
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.events import StopEvent
from llama_index.core.workflow.errors import WorkflowRuntimeError
from llama_index.core.workflow.handler import WorkflowHandler
from llama_index.core.workflow.workflow import Workflow, WorkflowMeta

DEFAULT_MAX_ITERATIONS = 20
DEFAULT_AGENT_NAME = "Agent"
DEFAULT_AGENT_DESCRIPTION = "An agent that can perform a task"
WORKFLOW_KWARGS = (
    "timeout",
    "verbose",
    "service_manager",
    "resource_manager",
    "num_concurrent_runs",
)


def get_default_llm() -> LLM:
    return Settings.llm


class BaseWorkflowAgentMeta(WorkflowMeta, ModelMetaclass):
    """Metaclass for BaseWorkflowAgent that properly combines WorkflowMeta, BaseModel's metaclass, and ABCMeta."""


class BaseWorkflowAgent(
    Workflow, BaseModel, PromptMixin, metaclass=BaseWorkflowAgentMeta
):
    """Base class for all agents, combining config and logic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default=DEFAULT_AGENT_NAME, description="The name of the agent")
    description: str = Field(
        default=DEFAULT_AGENT_DESCRIPTION,
        description="The description of what the agent does and is responsible for",
    )
    system_prompt: Optional[str] = Field(
        default=None, description="The system prompt for the agent"
    )
    tools: Optional[List[Union[BaseTool, Callable]]] = Field(
        default=None, description="The tools that the agent can use"
    )
    tool_retriever: Optional[ObjectRetriever] = Field(
        default=None,
        description="The tool retriever for the agent, can be provided instead of tools",
    )
    can_handoff_to: Optional[List[str]] = Field(
        default=None, description="The agent names that this agent can hand off to"
    )
    llm: LLM = Field(
        default_factory=get_default_llm, description="The LLM that the agent uses"
    )
    initial_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="The initial state of the agent, can be used by accessed under `await ctx.store.get('state')`",
    )
    state_prompt: Union[str, BasePromptTemplate] = Field(
        default=DEFAULT_STATE_PROMPT,
        description="The prompt to use to update the state of the agent",
        validate_default=True,
    )
    output_cls: Optional[Type[BaseModel]] = Field(
        description="Output class for the agent. If you set this field to a non-null value, `structured_output_fn` will be ignored.",
        default=None,
        exclude=True,
    )
    structured_output_fn: Optional[Callable[[List[ChatMessage]], Dict[str, Any]]] = (
        Field(
            description="Custom function to generate structured output from the agent's run. It has to take a list of ChatMessage instances (derived from the memory) and output a BaseModel subclass instance. If you set `output_cls` to a non-null value, this field will be ignored.",
            default=None,
        )
    )
    streaming: bool = Field(
        default=True,
        description="Whether to stream the agent's output to the event stream. Useful for long-running agents, but not every LLM will support streaming.",
    )

    def __init__(
        self,
        name: str = DEFAULT_AGENT_NAME,
        description: str = DEFAULT_AGENT_DESCRIPTION,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Union[BaseTool, Callable]]] = None,
        tool_retriever: Optional[ObjectRetriever] = None,
        can_handoff_to: Optional[List[str]] = None,
        llm: Optional[LLM] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        state_prompt: Optional[Union[str, BasePromptTemplate]] = None,
        output_cls: Optional[Type[BaseModel]] = None,
        structured_output_fn: Optional[Callable[[List[ChatMessage]], BaseModel]] = None,
        streaming: bool = True,
        timeout: Optional[float] = None,
        verbose: bool = False,
        **kwargs: Any,
    ):
        # Filter out workflow-specific kwargs
        workflow_kwargs = {k: v for k, v in kwargs.items() if k in WORKFLOW_KWARGS}
        model_kwargs = {k: v for k, v in kwargs.items() if k not in WORKFLOW_KWARGS}

        # Initialize BaseModel with the Pydantic fields
        if isinstance(state_prompt, str):
            state_prompt = PromptTemplate(state_prompt)
        elif state_prompt is None:
            state_prompt = DEFAULT_STATE_PROMPT

        if output_cls is not None and structured_output_fn is not None:
            structured_output_fn = None

        BaseModel.__init__(
            self,
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=tools,
            tool_retriever=tool_retriever,
            can_handoff_to=can_handoff_to,
            llm=llm or get_default_llm(),
            initial_state=initial_state or {},
            state_prompt=state_prompt,
            output_cls=output_cls,
            structured_output_fn=structured_output_fn,
            streaming=streaming,
            **model_kwargs,
        )

        # Initialize Workflow with workflow-specific parameters
        Workflow.__init__(self, timeout=timeout, verbose=verbose, **workflow_kwargs)

    @field_validator("tools", mode="before")
    def validate_tools(
        cls, v: Optional[Sequence[Union[BaseTool, Callable]]]
    ) -> Optional[Sequence[BaseTool]]:
        """
        Validate tools.

        If tools are not of type BaseTool, they will be converted to FunctionTools.
        This assumes the inputs are tools or callable functions.
        """
        if v is None:
            return None

        validated_tools: List[BaseTool] = []
        for tool in v:
            if not isinstance(tool, BaseTool):
                validated_tools.append(FunctionTool.from_defaults(tool))
            else:
                validated_tools.append(tool)

        for tool in validated_tools:
            if tool.metadata.name == "handoff":
                raise ValueError(
                    "'handoff' is a reserved tool name. Please use a different name."
                )

        return validated_tools  # type: ignore[return-value]

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        """Update prompts."""

    @abstractmethod
    async def take_step(
        self,
        ctx: Context,
        llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
        memory: BaseMemory,
    ) -> AgentOutput:
        """Take a single step with the agent."""

    @abstractmethod
    async def handle_tool_call_results(
        self, ctx: Context, results: List[ToolCallResult], memory: BaseMemory
    ) -> None:
        """Handle tool call results."""

    @abstractmethod
    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        """Finalize the agent's execution."""

    def _ensure_tools_are_async(
        self, tools: Sequence[BaseTool]
    ) -> Sequence[AsyncBaseTool]:
        """Ensure all tools are async."""
        return [adapt_to_async_tool(tool) for tool in tools]

    async def get_tools(
        self, input_str: Optional[str] = None
    ) -> Sequence[AsyncBaseTool]:
        """Get tools for the given agent."""
        tools = [*self.tools] if self.tools else []
        if self.tool_retriever is not None:
            retrieved_tools = await self.tool_retriever.aretrieve(input_str or "")
            tools.extend(retrieved_tools)

        return self._ensure_tools_are_async(cast(List[BaseTool], tools))

    async def _init_context(self, ctx: Context, ev: AgentWorkflowStartEvent) -> None:
        """Initialize the context once, if needed."""
        if not await ctx.store.get("memory", default=None):
            default_memory = ev.get("memory", default=None)
            default_memory = default_memory or ChatMemoryBuffer.from_defaults(
                llm=self.llm or Settings.llm
            )
            await ctx.store.set("memory", default_memory)
        if not await ctx.store.get("state", default=None):
            await ctx.store.set("state", self.initial_state.copy())

        if not await ctx.store.get("max_iterations", default=None):
            max_iterations = (
                ev.get("max_iterations", default=None) or DEFAULT_MAX_ITERATIONS
            )
            await ctx.store.set("max_iterations", max_iterations)

        # Reset the number of iterations
        await ctx.store.set("num_iterations", 0)

        # always set to false initially
        await ctx.store.set("formatted_input_with_state", False)

    async def _call_tool(
        self,
        ctx: Context,
        tool: AsyncBaseTool,
        tool_input: dict,
    ) -> ToolOutput:
        """Call the given tool with the given input."""
        try:
            if (
                isinstance(tool, FunctionTool)
                and tool.requires_context
                and tool.ctx_param_name is not None
            ):
                new_tool_input = {**tool_input}
                new_tool_input[tool.ctx_param_name] = ctx
                tool_output = await tool.acall(**new_tool_input)
            else:
                tool_output = await tool.acall(**tool_input)
        except Exception as e:
            # raise to wait
            waiting_for_event_exception = _get_waiting_for_event_exception()
            if waiting_for_event_exception and isinstance(
                e, waiting_for_event_exception
            ):
                raise
            tool_output = ToolOutput(
                content=str(e),
                tool_name=tool.metadata.get_name(),
                raw_input=tool_input,
                raw_output=str(e),
                is_error=True,
                exception=e,
            )

        return tool_output

    @step
    async def init_run(self, ctx: Context, ev: AgentWorkflowStartEvent) -> AgentInput:
        """Sets up the workflow and validates inputs."""
        await self._init_context(ctx, ev)

        user_msg: Optional[Union[str, ChatMessage]] = ev.get("user_msg")
        chat_history: Optional[List[ChatMessage]] = ev.get("chat_history", [])

        # Convert string user_msg to ChatMessage
        if isinstance(user_msg, str):
            user_msg = ChatMessage(role="user", content=user_msg)

        # Add messages to memory
        memory: BaseMemory = await ctx.store.get("memory")

        # First set chat history if it exists
        if chat_history:
            await memory.aset(chat_history)

        # Then add user message if it exists
        if user_msg:
            await memory.aput(user_msg)
            content_str = "\n".join(
                [
                    block.text
                    for block in user_msg.blocks
                    if isinstance(block, TextBlock)
                ]
            )
            await ctx.store.set("user_msg_str", content_str)
        elif chat_history and not all(
            message.role == "system" for message in chat_history
        ):
            # If no user message, use the last message from chat history as user_msg_str
            user_hist: List[ChatMessage] = [
                msg for msg in chat_history if msg.role == "user"
            ]
            content_str = "\n".join(
                [
                    block.text
                    for block in user_hist[-1].blocks
                    if isinstance(block, TextBlock)
                ]
            )
            await ctx.store.set("user_msg_str", content_str)
        else:
            raise ValueError("Must provide either user_msg or chat_history")

        # Get all messages from memory
        input_messages = await memory.aget()

        # send to the current agent
        return AgentInput(input=input_messages, current_agent_name=self.name)

    @step
    async def setup_agent(self, ctx: Context, ev: AgentInput) -> AgentSetup:
        """Main agent handling logic."""
        llm_input = [*ev.input]

        if self.system_prompt:
            llm_input = [
                ChatMessage(role="system", content=self.system_prompt),
                *llm_input,
            ]

        state = await ctx.store.get("state", default=None)
        formatted_input_with_state = await ctx.store.get(
            "formatted_input_with_state", default=False
        )
        if state and not formatted_input_with_state:
            # update last message with current state
            for block in llm_input[-1].blocks[::-1]:
                if isinstance(block, TextBlock):
                    block.text = self.state_prompt.format(state=state, msg=block.text)
                    break
            await ctx.store.set("formatted_input_with_state", True)

        return AgentSetup(
            input=llm_input,
            current_agent_name=ev.current_agent_name,
        )

    @step
    async def run_agent_step(self, ctx: Context, ev: AgentSetup) -> AgentOutput:
        """Run the agent."""
        memory: BaseMemory = await ctx.store.get("memory")
        user_msg_str = await ctx.store.get("user_msg_str")
        tools = await self.get_tools(user_msg_str or "")

        agent_output = await self.take_step(
            ctx,
            ev.input,
            tools,
            memory,
        )

        ctx.write_event_to_stream(agent_output)
        return agent_output

    @step
    async def parse_agent_output(
        self, ctx: Context, ev: AgentOutput
    ) -> Union[StopEvent, AgentInput, ToolCall, None]:
        max_iterations = await ctx.store.get(
            "max_iterations", default=DEFAULT_MAX_ITERATIONS
        )
        num_iterations = await ctx.store.get("num_iterations", default=0)
        num_iterations += 1
        await ctx.store.set("num_iterations", num_iterations)

        if num_iterations >= max_iterations:
            raise WorkflowRuntimeError(
                f"Max iterations of {max_iterations} reached! Either something went wrong, or you can "
                "increase the max iterations with `.run(.., max_iterations=...)`"
            )

        memory: BaseMemory = await ctx.store.get("memory")

        if ev.retry_messages:
            # Retry with the given messages to let the LLM fix potential errors
            history = await memory.aget()
            user_msg_str = await ctx.store.get("user_msg_str")

            return AgentInput(
                input=[
                    *history,
                    ChatMessage(role="user", content=user_msg_str),
                    *ev.retry_messages,
                ],
                current_agent_name=self.name,
            )

        if not ev.tool_calls:
            # important: messages should always be fetched after calling finalize, otherwise they do not contain the agent's response
            output = await self.finalize(ctx, ev, memory)
            messages = await memory.aget()
            cur_tool_calls: List[ToolCallResult] = await ctx.store.get(
                "current_tool_calls", default=[]
            )
            output.tool_calls.extend(cur_tool_calls)  # type: ignore

            if self.structured_output_fn is not None:
                try:
                    if inspect.iscoroutinefunction(self.structured_output_fn):
                        output.structured_response = await self.structured_output_fn(
                            messages
                        )
                    else:
                        output.structured_response = cast(
                            Dict[str, Any], self.structured_output_fn(messages)
                        )
                    ctx.write_event_to_stream(
                        AgentStreamStructuredOutput(output=output.structured_response)
                    )
                except Exception as e:
                    warnings.warn(
                        f"There was a problem with the generation of the structured output: {e}"
                    )
            if self.output_cls is not None:
                try:
                    llm_input = [*messages]
                    if self.system_prompt:
                        llm_input = [
                            ChatMessage(role="system", content=self.system_prompt),
                            *llm_input,
                        ]
                    output.structured_response = await generate_structured_response(
                        messages=llm_input, llm=self.llm, output_cls=self.output_cls
                    )
                    ctx.write_event_to_stream(
                        AgentStreamStructuredOutput(output=output.structured_response)
                    )
                except Exception as e:
                    warnings.warn(
                        f"There was a problem with the generation of the structured output: {e}"
                    )

            await ctx.store.set("current_tool_calls", [])

            return StopEvent(result=output)

        await ctx.store.set("num_tool_calls", len(ev.tool_calls))

        for tool_call in ev.tool_calls:
            ctx.send_event(
                ToolCall(
                    tool_name=tool_call.tool_name,
                    tool_kwargs=tool_call.tool_kwargs,
                    tool_id=tool_call.tool_id,
                )
            )

        return None

    @step
    async def call_tool(self, ctx: Context, ev: ToolCall) -> ToolCallResult:
        """Calls the tool and handles the result."""
        ctx.write_event_to_stream(
            ToolCall(
                tool_name=ev.tool_name,
                tool_kwargs=ev.tool_kwargs,
                tool_id=ev.tool_id,
            )
        )

        tools = await self.get_tools(ev.tool_name)
        tools_by_name = {tool.metadata.name: tool for tool in tools}
        if ev.tool_name not in tools_by_name:
            tool = None
            result = ToolOutput(
                content=f"Tool {ev.tool_name} not found. Please select a tool that is available.",
                tool_name=ev.tool_name,
                raw_input=ev.tool_kwargs,
                raw_output=None,
                is_error=True,
            )
        else:
            tool = tools_by_name[ev.tool_name]
            result = await self._call_tool(ctx, tool, ev.tool_kwargs)

        result_ev = ToolCallResult(
            tool_name=ev.tool_name,
            tool_kwargs=ev.tool_kwargs,
            tool_id=ev.tool_id,
            tool_output=result,
            return_direct=tool.metadata.return_direct if tool else False,
        )

        ctx.write_event_to_stream(result_ev)
        return result_ev

    @step
    async def aggregate_tool_results(
        self, ctx: Context, ev: ToolCallResult
    ) -> Union[AgentInput, StopEvent, None]:
        """Aggregate tool results and return the next agent input."""
        num_tool_calls = await ctx.store.get("num_tool_calls", default=0)
        if num_tool_calls == 0:
            raise ValueError("No tool calls found, cannot aggregate results.")

        tool_call_results: list[ToolCallResult] = ctx.collect_events(  # type: ignore
            ev, expected=[ToolCallResult] * num_tool_calls
        )
        if not tool_call_results:
            return None

        memory: BaseMemory = await ctx.store.get("memory")

        # track tool calls made during a .run() call
        cur_tool_calls: List[ToolCallResult] = await ctx.store.get(
            "current_tool_calls", default=[]
        )
        cur_tool_calls.extend(tool_call_results)
        await ctx.store.set("current_tool_calls", cur_tool_calls)

        await self.handle_tool_call_results(ctx, tool_call_results, memory)

        if any(
            tool_call_result.return_direct and not tool_call_result.tool_output.is_error
            for tool_call_result in tool_call_results
        ):
            # if any tool calls return directly and it's not an error tool call, take the first one
            return_direct_tool = next(
                tool_call_result
                for tool_call_result in tool_call_results
                if tool_call_result.return_direct
                and not tool_call_result.tool_output.is_error
            )

            # always finalize the agent, even if we're just handing off
            result = AgentOutput(
                response=ChatMessage(
                    role="assistant",
                    content=return_direct_tool.tool_output.content or "",
                ),
                tool_calls=[
                    ToolSelection(
                        tool_id=t.tool_id,
                        tool_name=t.tool_name,
                        tool_kwargs=t.tool_kwargs,
                    )
                    for t in cur_tool_calls
                ],
                raw=return_direct_tool.tool_output.raw_output,
                current_agent_name=self.name,
            )
            result = await self.finalize(ctx, result, memory)
            # we don't want to stop the system if we're just handing off
            if return_direct_tool.tool_name != "handoff":
                await ctx.store.set("current_tool_calls", [])
                return StopEvent(result=result)

        user_msg_str = await ctx.store.get("user_msg_str")
        input_messages = await memory.aget(input=user_msg_str)

        return AgentInput(input=input_messages, current_agent_name=self.name)

    def run(
        self,
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        ctx: Optional[Context] = None,
        max_iterations: Optional[int] = None,
        start_event: Optional[AgentWorkflowStartEvent] = None,
        **kwargs: Any,
    ) -> WorkflowHandler:
        # Detect if hitl is needed
        if ctx is not None and ctx.is_running:
            return super().run(
                ctx=ctx,
                **kwargs,
            )
        else:
            start_event = start_event or AgentWorkflowStartEvent(
                user_msg=user_msg,
                chat_history=chat_history,
                memory=memory,
                max_iterations=max_iterations,
                **kwargs,
            )
            return super().run(
                start_event=start_event,
                ctx=ctx,
            )


@functools.lru_cache(maxsize=1)
def _get_waiting_for_event_exception() -> Optional[Type[Exception]]:
    try:
        # Special exception introduced in workflows 2.9.0 as a way to fully pause waiting steps.
        # If it exists, check for it and re-raise
        from workflows.runtime.types.results import WaitingForEvent

        return WaitingForEvent
    except ImportError:
        return None
