import uuid
from typing import Any, Dict, List, Optional, Union, cast

from llama_index.core.agent.multi_agent.agent_config import AgentConfig, AgentMode
from llama_index.core.agent.multi_agent.workflow_events import (
    ToolCall,
    ToolCallResult,
    AgentInput,
    AgentSetup,
    AgentStream,
    AgentOutput,
)
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.tools import (
    BaseTool,
    AsyncBaseTool,
    ToolOutput,
    adapt_to_async_tool,
)
from llama_index.core.workflow import (
    Context,
    FunctionToolWithContext,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.core.settings import Settings


DEFAULT_HANDOFF_PROMPT = """Useful for handing off to another agent.
If you are currently not equipped to handle the user's request, or another agent is better suited to handle the request, please hand off to the appropriate agent.

Currently available agents:
{agent_info}
"""


async def handoff(ctx: Context, to_agent: str, reason: str) -> str:
    """Handoff control of that chat to the given agent."""
    agent_configs = await ctx.get("agent_configs")
    current_agent = await ctx.get("current_agent")
    if to_agent not in agent_configs:
        valid_agents = ", ".join([x for x in agent_configs if x != current_agent])
        return f"Agent {to_agent} not found. Please select a valid agent to hand off to. Valid agents: {valid_agents}"

    await ctx.set("current_agent", to_agent)
    return f"Handed off to {to_agent} because: {reason}"


class MultiAgentWorkflow(Workflow):
    """A workflow for managing multiple agents with handoffs."""

    def __init__(
        self,
        agent_configs: List[AgentConfig],
        initial_state: Optional[Dict] = None,
        handoff_prompt: Optional[Union[str, BasePromptTemplate]] = None,
        state_prompt: Optional[Union[str, BasePromptTemplate]] = None,
        timeout: Optional[float] = None,
        **workflow_kwargs: Any,
    ):
        super().__init__(timeout=timeout, **workflow_kwargs)
        if not agent_configs:
            raise ValueError("At least one agent config must be provided")

        self.agent_configs = {cfg.name: cfg for cfg in agent_configs}
        only_one_root_agent = sum(cfg.is_entrypoint_agent for cfg in agent_configs) == 1
        if not only_one_root_agent:
            raise ValueError("Exactly one root agent must be provided")

        self.root_agent = next(
            cfg.name for cfg in agent_configs if cfg.is_entrypoint_agent
        )

        self.initial_state = initial_state or {}

        self.handoff_prompt = handoff_prompt or DEFAULT_HANDOFF_PROMPT
        if isinstance(self.handoff_prompt, str):
            self.handoff_prompt = PromptTemplate(self.handoff_prompt)
            if "{agent_info}" not in self.handoff_prompt.template:
                raise ValueError("Handoff prompt must contain {agent_info}")

        self.state_prompt = state_prompt
        if isinstance(self.state_prompt, str):
            self.state_prompt = PromptTemplate(self.state_prompt)
            if (
                "{state}" not in self.state_prompt.template
                or "{msg}" not in self.state_prompt.template
            ):
                raise ValueError("State prompt must contain {state} and {msg}")

    def _ensure_tools_are_async(self, tools: List[BaseTool]) -> List[AsyncBaseTool]:
        """Ensure all tools are async."""
        return [adapt_to_async_tool(tool) for tool in tools]

    def _get_handoff_tool(self, current_agent_config: AgentConfig) -> AsyncBaseTool:
        """Creates a handoff tool for the given agent."""
        agent_info = {cfg.name: cfg.description for cfg in self.agent_configs.values()}

        # Filter out agents that the current agent cannot handoff to
        configs_to_remove = []
        for name in agent_info:
            if name == current_agent_config.name:
                configs_to_remove.append(name)
            elif (
                current_agent_config.can_handoff_to is not None
                and name not in current_agent_config.can_handoff_to
            ):
                configs_to_remove.append(name)

        for name in configs_to_remove:
            agent_info.pop(name)

        fn_tool_prompt = self.handoff_prompt.format(agent_info=str(agent_info))
        return FunctionToolWithContext.from_defaults(
            async_fn=handoff, description=fn_tool_prompt, return_direct=True
        )

    async def _init_context(self, ctx: Context, ev: StartEvent) -> None:
        """Initialize the context once, if needed."""
        if not await ctx.get("memory", default=None):
            default_memory = ev.get("memory", default=None)
            default_memory = default_memory or ChatMemoryBuffer.from_defaults(
                llm=self.agent_configs[self.root_agent].llm or Settings.llm
            )
            await ctx.set("memory", default_memory)
        if not await ctx.get("agent_configs", default=None):
            await ctx.set("agent_configs", self.agent_configs)
        if not await ctx.get("state", default=None):
            await ctx.set("state", self.initial_state)
        if not await ctx.get("current_agent", default=None):
            await ctx.set("current_agent", self.root_agent)

    async def _call_tool(
        self,
        ctx: Context,
        tool: AsyncBaseTool,
        tool_input: dict,
    ) -> ToolOutput:
        """Call the given tool with the given input."""
        try:
            if isinstance(tool, FunctionToolWithContext):
                tool_output = await tool.acall(ctx=ctx, **tool_input)
            else:
                tool_output = await tool.acall(**tool_input)
        except Exception as e:
            tool_output = ToolOutput(
                content=str(e),
                tool_name=tool.metadata.name,
                raw_input=tool_input,
                raw_output=str(e),
                is_error=True,
            )

        return tool_output

    async def _handle_react_tool_call(
        self, ctx: Context, results: List[ToolCallResult]
    ) -> None:
        """Adds to the react reasoning list."""
        current_reasoning: list[BaseReasoningStep] = await ctx.get(
            "current_reasoning", default=[]
        )
        for tool_call_result in results:
            # don't add handoff tool calls to reasoning
            if tool_call_result.tool_name == "handoff":
                continue

            current_reasoning.append(
                ObservationReasoningStep(
                    observation=str(tool_call_result.tool_output.content),
                    return_direct=tool_call_result.return_direct,
                )
            )

            if tool_call_result.return_direct:
                current_reasoning.append(
                    ResponseReasoningStep(
                        thought=current_reasoning[-1].observation,
                        response=current_reasoning[-1].observation,
                        is_streaming=False,
                    )
                )
                break

        await ctx.set("current_reasoning", current_reasoning)

    async def _handle_function_tool_call(
        self, ctx: Context, results: List[ToolCallResult]
    ) -> None:
        """Adds to memory."""
        memory: BaseMemory = await ctx.get("memory")
        for tool_call_result in results:
            # don't add handoff tool calls to memory
            if tool_call_result.tool_name == "handoff":
                continue

            await memory.aput(
                ChatMessage(
                    role="tool",
                    content=str(tool_call_result.tool_output.content),
                    additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                )
            )

            if tool_call_result.return_direct:
                await memory.aput(
                    ChatMessage(
                        role="assistant",
                        content=str(tool_call_result.tool_output.content),
                        additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                    )
                )
                break

        await ctx.set("memory", memory)

    async def _call_function_calling_agent(
        self,
        ctx: Context,
        llm: FunctionCallingLLM,
        llm_input: List[ChatMessage],
        tools: List[AsyncBaseTool],
    ) -> AgentOutput:
        """Call the LLM as a function calling agent."""
        memory: BaseMemory = await ctx.get("memory")
        current_agent = await ctx.get("current_agent")

        current_llm_input = [*llm_input]
        response = await llm.astream_chat_with_tools(
            tools, chat_history=current_llm_input, allow_parallel_tool_calls=True
        )
        async for r in response:
            tool_calls = llm.get_tool_calls_from_response(
                r, error_on_no_tool_call=False
            )
            ctx.write_event_to_stream(
                AgentStream(
                    delta=r.delta or "",
                    tool_calls=tool_calls or [],
                    raw_response=r.raw,
                    current_agent=current_agent,
                )
            )

        tool_calls = llm.get_tool_calls_from_response(r, error_on_no_tool_call=False)

        # only add to memory if we didn't select the handoff tool
        if not any(tool_call.tool_name == "handoff" for tool_call in tool_calls):
            current_llm_input.append(r.message)
            await memory.aput(r.message)
            await ctx.set("memory", memory)

        return AgentOutput(
            response=r.message.content,
            tool_calls=tool_calls or [],
            raw_response=r.raw,
            current_agent=current_agent,
        )

    async def _call_react_agent(
        self,
        ctx: Context,
        llm: LLM,
        llm_input: List[ChatMessage],
        tools: List[AsyncBaseTool],
    ) -> AgentOutput:
        """Call the LLM as a react agent."""
        memory: BaseMemory = await ctx.get("memory")
        current_agent = await ctx.get("current_agent")

        # remove system prompt, since the react prompt will be combined with it
        if llm_input[0].role == "system":
            system_prompt = llm_input[0].content or ""
            llm_input = llm_input[1:]
        else:
            system_prompt = ""

        output_parser = ReActOutputParser()
        react_chat_formatter = ReActChatFormatter(context=system_prompt)

        # Format initial chat input
        current_reasoning: list[BaseReasoningStep] = await ctx.get(
            "current_reasoning", default=[]
        )
        input_chat = react_chat_formatter.format(
            tools,
            chat_history=llm_input,
            current_reasoning=current_reasoning,
        )

        # Initial LLM call
        response = await llm.astream_chat(input_chat)
        async for r in response:
            ctx.write_event_to_stream(
                AgentStream(
                    delta=r.delta or "",
                    tool_calls=[],
                    raw_response=r.raw,
                    current_agent=current_agent,
                )
            )

        # Parse reasoning step and check if done
        message_content = r.message.content
        if not message_content:
            raise ValueError("Got empty message")

        try:
            reasoning_step = output_parser.parse(message_content, is_streaming=False)
        except ValueError as e:
            # If we can't parse the output, return an error message
            error_msg = f"Error: Could not parse output. Please follow the thought-action-input format. Try again. Details: {e!s}"

            await memory.aput(r.message.content)
            await memory.aput(ChatMessage(role="user", content=error_msg))
            await ctx.set("memory", memory)

            return AgentOutput(
                response=r.message.content,
                tool_calls=[],
                raw_response=r.raw,
                current_agent=current_agent,
            )

        # add to reasoning if not a handoff
        if hasattr(reasoning_step, "action") and reasoning_step.action != "handoff":
            current_reasoning.append(reasoning_step)
            await ctx.set("current_reasoning", current_reasoning)

        # If response step, we're done
        if reasoning_step.is_done:
            return AgentOutput(
                response=r.message.content,
                tool_calls=[],
                raw_response=r.raw,
                current_agent=current_agent,
            )

        reasoning_step = cast(ActionReasoningStep, reasoning_step)
        if not isinstance(reasoning_step, ActionReasoningStep):
            raise ValueError(f"Expected ActionReasoningStep, got {reasoning_step}")

        # Create tool call
        tool_calls = [
            ToolSelection(
                tool_id=str(uuid.uuid4()),
                tool_name=reasoning_step.action,
                tool_kwargs=reasoning_step.action_input,
            )
        ]

        return AgentOutput(
            response=r.message.content,
            tool_calls=tool_calls,
            raw_response=r.raw,
            current_agent=current_agent,
        )

    async def _call_llm(
        self,
        ctx: Context,
        llm: LLM,
        llm_input: List[ChatMessage],
        tools: List[AsyncBaseTool],
        agent_config: AgentConfig,
    ) -> AgentOutput:
        """Call the LLM with the given input and tools."""
        if agent_config.get_mode() == AgentMode.REACT:
            return await self._call_react_agent(ctx, llm, llm_input, tools)
        elif agent_config.get_mode() == AgentMode.FUNCTION:
            return await self._call_function_calling_agent(ctx, llm, llm_input, tools)
        else:
            raise ValueError(f"Invalid agent mode: {agent_config.get_mode()}")

    async def _finalize_function_calling_agent(self, ctx: Context) -> None:
        """Finalizes the function calling agent.

        This is a no-op for the function calling agent, since we've been writing to memory as we go.
        """

    async def _finalize_react_agent(self, ctx: Context) -> None:
        """Finalizes the react agent by writing the current reasoning to memory."""
        memory: BaseMemory = await ctx.get("memory")
        current_reasoning: list[BaseReasoningStep] = await ctx.get(
            "current_reasoning", default=[]
        )

        reasoning_str = "\n".join([x.get_content() for x in current_reasoning])
        reasoning_msg = ChatMessage(role="assistant", content=reasoning_str)

        await memory.aput(reasoning_msg)
        await ctx.set("memory", memory)
        await ctx.set("current_reasoning", [])

    @step
    async def init_run(self, ctx: Context, ev: StartEvent) -> AgentInput:
        """Sets up the workflow and validates inputs."""
        await self._init_context(ctx, ev)

        user_msg = ev.get("user_msg")
        chat_history = ev.get("chat_history")
        if user_msg and chat_history:
            raise ValueError("Cannot provide both user_msg and chat_history")

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role="user", content=user_msg)

        await ctx.set("user_msg_str", user_msg.content)

        # Add messages to memory
        memory: BaseMemory = await ctx.get("memory")
        if user_msg:
            await memory.aput(user_msg)
            input_messages = memory.get(input=user_msg.content)

            # Add the state to the user message if it exists and if requested
            current_state = await ctx.get("state")
            if self.state_prompt and current_state:
                user_msg.content = self.state_prompt.format(
                    state=current_state, msg=user_msg.content
                )

            await memory.aput(user_msg)
        else:
            memory.set(chat_history)
            input_messages = memory.get()

        # send to the current agent
        current_agent = await ctx.get("current_agent")
        return AgentInput(input=input_messages, current_agent=current_agent)

    @step
    async def setup_agent(self, ctx: Context, ev: AgentInput) -> AgentSetup:
        """Main agent handling logic."""
        agent_config: AgentConfig = self.agent_configs[ev.current_agent]
        llm_input = ev.input

        # Set up the tools
        tools = list(agent_config.tools or [])
        if agent_config.tool_retriever:
            retrieved_tools = await agent_config.tool_retriever.aretrieve(
                llm_input[-1].content or str(llm_input)
            )
            tools.extend(retrieved_tools)

        if agent_config.can_handoff_to or agent_config.can_handoff_to is None:
            handoff_tool = self._get_handoff_tool(agent_config)
            tools.append(handoff_tool)

        tools = self._ensure_tools_are_async(tools)

        ctx.write_event_to_stream(
            AgentInput(input=llm_input, current_agent=ev.current_agent)
        )

        if agent_config.system_prompt:
            llm_input = [
                ChatMessage(role="system", content=agent_config.system_prompt),
                *llm_input,
            ]

        await ctx.set("tools_by_name", {tool.metadata.name: tool for tool in tools})

        return AgentSetup(
            input=llm_input,
            current_agent=ev.current_agent,
            current_config=agent_config,
            tools=tools,
        )

    @step
    async def run_agent_step(self, ctx: Context, ev: AgentSetup) -> AgentOutput:
        """Run the agent."""
        agent_config = ev.current_config
        llm = agent_config.llm or Settings.llm

        agent_output = await self._call_llm(
            ctx,
            llm,
            ev.input,
            ev.tools,
            agent_config,
        )
        ctx.write_event_to_stream(agent_output)

        return agent_output

    @step
    async def parse_agent_output(
        self, ctx: Context, ev: AgentOutput
    ) -> StopEvent | ToolCall | None:
        if not ev.tool_calls:
            agent_configs = self.agent_configs
            current_config = agent_configs[ev.current_agent]
            if current_config.get_mode() == AgentMode.REACT:
                await self._finalize_react_agent(ctx)
            elif current_config.get_mode() == AgentMode.FUNCTION:
                await self._finalize_function_calling_agent(ctx)
            else:
                raise ValueError(f"Invalid agent mode: {current_config.get_mode()}")

            return StopEvent(result=ev)

        await ctx.set("num_tool_calls", len(ev.tool_calls))

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
        ctx.write_event_to_stream(ev)

        tools_by_name: dict[str, AsyncBaseTool] = await ctx.get("tools_by_name")
        if ev.tool_name not in tools_by_name:
            result = ToolOutput(
                content=f"Tool {ev.tool_name} not found. Please select a tool that is available.",
                tool_name=ev.tool_name,
                raw_input=ev.tool_kwargs,
                raw_output=None,
                is_error=True,
                return_direct=False,
            )
        else:
            tool = tools_by_name[ev.tool_name]
            result = await self._call_tool(ctx, tool, ev.tool_kwargs)

        result_ev = ToolCallResult(
            tool_name=ev.tool_name,
            tool_kwargs=ev.tool_kwargs,
            tool_id=ev.tool_id,
            tool_output=result,
            return_direct=tool.metadata.return_direct,
        )

        ctx.write_event_to_stream(result_ev)
        return result_ev

    @step
    async def aggregate_tool_results(
        self, ctx: Context, ev: ToolCallResult
    ) -> AgentInput | StopEvent | None:
        """Aggregate tool results and return the next agent input."""
        num_tool_calls = await ctx.get("num_tool_calls", default=0)
        if num_tool_calls == 0:
            raise ValueError("No tool calls found, cannot aggregate results.")

        tool_call_results: list[ToolCallResult] = ctx.collect_events(
            ev, expected=[ToolCallResult] * num_tool_calls
        )
        if not tool_call_results:
            return None

        current_agent = await ctx.get("current_agent")
        current_config: AgentConfig = self.agent_configs[current_agent]

        if current_config.get_mode() == AgentMode.REACT:
            await self._handle_react_tool_call(ctx, tool_call_results)
        elif current_config.get_mode() == AgentMode.FUNCTION:
            await self._handle_function_tool_call(ctx, tool_call_results)
        else:
            raise ValueError(f"Invalid agent mode: {current_config.get_mode()}")

        if any(
            tool_call_result.return_direct for tool_call_result in tool_call_results
        ):
            # if any tool calls return directly, take the first one
            return_direct_tool = next(
                tool_call_result
                for tool_call_result in tool_call_results
                if tool_call_result.return_direct
            )

            current_config = self.agent_configs[current_agent]
            if current_config.get_mode() == AgentMode.REACT:
                await self._finalize_react_agent(ctx)
            elif current_config.get_mode() == AgentMode.FUNCTION:
                await self._finalize_function_calling_agent(ctx)
            else:
                raise ValueError(f"Invalid agent mode: {current_config.get_mode()}")

            # we don't want to stop the system if we're just handing off
            if return_direct_tool.tool_name != "handoff":
                return StopEvent(
                    result=AgentOutput(
                        response=return_direct_tool.tool_output.content,
                        tool_calls=[
                            ToolSelection(
                                tool_id=t.tool_id,
                                tool_name=t.tool_name,
                                tool_kwargs=t.tool_kwargs,
                            )
                            for t in tool_call_results
                        ],
                        raw_response=return_direct_tool.tool_output.raw_output,
                        current_agent=current_agent,
                    )
                )

        user_msg_str = await ctx.get("user_msg_str")
        memory: BaseMemory = await ctx.get("memory")
        input_messages = memory.get(input=user_msg_str)

        # get this again, in case it changed
        current_agent = await ctx.get("current_agent")

        return AgentInput(input=input_messages, current_agent=current_agent)
