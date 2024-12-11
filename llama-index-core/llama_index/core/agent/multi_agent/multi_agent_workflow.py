import asyncio
from typing import Any, Dict, List, Optional, Union, cast

from llama_index.core.agent.multi_agent.agent_config import AgentConfig, AgentMode
from llama_index.core.agent.multi_agent.workflow_events import (
    HandoffEvent,
    ToolCall,
    AgentInput,
    AgentSetup,
    AgentStream,
    AgentOutput,
)
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.tools import (
    BaseTool,
    AsyncBaseTool,
    FunctionTool,
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
If you are currently not equipped to handle the user's request, please hand off to the appropriate agent.

Currently available agents:
{agent_info}
"""


async def handoff(to_agent: str, reason: str) -> HandoffEvent:
    """Handoff to the given agent."""
    return f"Handed off to {to_agent} because: {reason}"


class MultiAgentWorkflow(Workflow):
    """A workflow for managing multiple agents with handoffs."""

    def __init__(
        self,
        agent_configs: List[AgentConfig],
        initial_state: Optional[Dict] = None,
        memory: Optional[BaseMemory] = None,
        handoff_prompt: Optional[Union[str, BasePromptTemplate]] = None,
        state_prompt: Optional[Union[str, BasePromptTemplate]] = None,
        timeout: Optional[float] = None,
        **workflow_kwargs: Any,
    ):
        super().__init__(timeout=timeout, **workflow_kwargs)
        if not agent_configs:
            raise ValueError("At least one agent config must be provided")

        self.agent_configs = {cfg.name: cfg for cfg in agent_configs}
        only_one_root_agent = sum(cfg.is_root_agent for cfg in agent_configs) == 1
        if not only_one_root_agent:
            raise ValueError("Exactly one root agent must be provided")

        self.root_agent = next(cfg.name for cfg in agent_configs if cfg.is_root_agent)

        self.initial_state = initial_state or {}
        self.memory = memory or ChatMemoryBuffer.from_defaults(
            llm=agent_configs[0].llm or Settings.llm
        )

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
        return FunctionTool.from_defaults(async_fn=handoff, description=fn_tool_prompt)

    async def _init_context(self, ctx: Context) -> None:
        """Initialize the context once, if needed."""
        if not await ctx.get("memory", default=None):
            await ctx.set("memory", self.memory)
        if not await ctx.get("agent_configs", default=None):
            await ctx.set("agent_configs", self.agent_configs)
        if not await ctx.get("current_state", default=None):
            await ctx.set("current_state", self.initial_state)
        if not await ctx.get("current_agent", default=None):
            await ctx.set("current_agent", self.root_agent)

    async def _call_tool(
        self, ctx: Context, tool: AsyncBaseTool, tool_input: dict
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

        ctx.write_event_to_stream(
            ToolCall(
                tool_name=tool.metadata.name,
                tool_kwargs=tool_input,
                tool_output=tool_output.content,
            )
        )

        return tool_output

    async def _call_function_calling_agent(
        self,
        ctx: Context,
        llm: FunctionCallingLLM,
        llm_input: List[ChatMessage],
        tools: List[AsyncBaseTool],
    ) -> AgentOutput:
        """Call the LLM as a function calling agent."""
        memory: BaseMemory = await ctx.get("memory")
        tools_by_name = {tool.metadata.name: tool for tool in tools}

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
                )
            )

        current_llm_input.append(r.message)
        await memory.aput(r.message)
        tool_calls = llm.get_tool_calls_from_response(r, error_on_no_tool_call=False)

        all_tool_results = []
        while tool_calls:
            tool_results: List[ToolOutput] = []
            tool_ids: List[str] = []
            jobs = []
            should_return_direct = False
            for tool_call in tool_calls:
                tool_ids.append(tool_call.tool_call_id)
                if tool_call.tool_name not in tools_by_name:
                    tool_results.append(
                        ToolOutput(
                            content=f"Tool {tool_call.tool_name} not found. Please select a tool that is available.",
                            tool_name=tool_call.tool_name,
                            raw_input=tool_call.tool_kwargs,
                            raw_output=None,
                            is_error=True,
                        )
                    )
                else:
                    tool = tools_by_name[tool_call.tool_name]
                    if tool.metadata.return_direct:
                        should_return_direct = True

                    job = self._call_tool(ctx, tool, tool_call.tool_kwargs)
                    jobs.append(job)

            tool_results.extend(await asyncio.gather(*jobs))
            all_tool_results.extend(tool_results)
            tool_messages = [
                ChatMessage(
                    role="tool",
                    content=str(result),
                    additional_kwargs={"tool_call_id": tool_id},
                )
                for result, tool_id in zip(tool_results, tool_ids)
            ]

            for tool_message in tool_messages:
                await memory.aput(tool_message)

            if should_return_direct:
                return AgentOutput(
                    response=tool_results[0].content,
                    tool_outputs=all_tool_results,
                    raw_response=None,
                )

            current_llm_input.extend(tool_messages)
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
                    )
                )

            current_llm_input.append(r.message)
            await memory.aput(r.message)
            tool_calls = llm.get_tool_calls_from_response(
                r, error_on_no_tool_call=False
            )

        return AgentOutput(
            response=r.message.content,
            tool_outputs=all_tool_results,
            raw_response=r.raw,
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

        # remove system prompt, since the react prompt will be combined with it
        if llm_input[0].role == "system":
            system_prompt = llm_input[0].content or ""
            llm_input = llm_input[1:]
        else:
            system_prompt = ""

        output_parser = ReActOutputParser()
        react_chat_formatter = ReActChatFormatter(context=system_prompt)

        # Format initial chat input
        current_reasoning = []
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
                )
            )

        await memory.aput(r.message)

        # Parse reasoning step and check if done
        message_content = r.message.content
        if not message_content:
            raise ValueError("Got empty message")

        try:
            reasoning_step = output_parser.parse(message_content, is_streaming=False)
        except ValueError as e:
            # If we can't parse the output, return an error message
            error_msg = f"Error: Could not parse output. Please follow the thought-action-input format. Try again. Details: {e!s}"
            return AgentOutput(
                response=error_msg,
                tool_outputs=[],
                raw_response=r.raw,
            )

        # If response step, we're done
        all_tool_outputs = []
        if reasoning_step.is_done:
            current_reasoning.append(reasoning_step)

            latest_react_messages = react_chat_formatter.format(
                tools,
                chat_history=llm_input,
                current_reasoning=current_reasoning,
            )
            for msg in latest_react_messages:
                await memory.aput(msg)

            response = (
                reasoning_step.response
                if hasattr(reasoning_step, "response")
                else reasoning_step.get_content()
            )
            return AgentOutput(
                response=response,
                tool_outputs=all_tool_outputs,
                raw_response=r.raw,
            )

        # Otherwise process action step
        while True:
            current_reasoning.append(reasoning_step)

            reasoning_step = cast(ActionReasoningStep, reasoning_step)
            if not isinstance(reasoning_step, ActionReasoningStep):
                raise ValueError(f"Expected ActionReasoningStep, got {reasoning_step}")

            # Call tool
            tools_by_name = {tool.metadata.name: tool for tool in tools}
            tool = None
            if reasoning_step.action not in tools_by_name:
                tool_output = ToolOutput(
                    content=f"Error: No such tool named `{reasoning_step.action}`.",
                    tool_name=reasoning_step.action,
                    raw_input={"kwargs": reasoning_step.action_input},
                    raw_output=None,
                    is_error=True,
                )
            else:
                tool = tools_by_name[reasoning_step.action]
                tool_output = await self._call_tool(
                    ctx, tool, reasoning_step.action_input
                )
                all_tool_outputs.append(tool_output)

            # Add observation to chat history
            current_reasoning.append(
                ObservationReasoningStep(
                    observation=str(tool_output),
                    return_direct=tool.metadata.return_direct,
                )
            )

            if tool and tool.metadata.return_direct:
                current_reasoning.append(
                    ResponseReasoningStep(response=tool_output.content)
                )
                latest_react_messages = react_chat_formatter.format(
                    tools,
                    chat_history=llm_input,
                    current_reasoning=current_reasoning,
                )
                for msg in latest_react_messages:
                    await memory.aput(msg)

                return AgentOutput(
                    response=tool_output.content,
                    tool_outputs=all_tool_outputs,
                    raw_response=r.raw,
                )

            # Get next action from LLM
            input_chat = react_chat_formatter.format(
                tools,
                chat_history=llm_input,
                current_reasoning=current_reasoning,
            )
            response = await llm.astream_chat(input_chat)
            async for r in response:
                ctx.write_event_to_stream(
                    AgentStream(
                        delta=r.delta or "",
                        tool_calls=[],
                        raw_response=r.raw,
                    )
                )

            await memory.aput(r.message)

            # Parse next reasoning step
            message_content = r.message.content
            if not message_content:
                raise ValueError("Got empty message")

            try:
                reasoning_step = output_parser.parse(
                    message_content, is_streaming=False
                )
            except ValueError as e:
                # If we can't parse the output, return an error message
                error_msg = f"Error: Could not parse output. Please follow the thought-action-input format. Try again. Details: {e!s}"

                current_reasoning.append(ResponseReasoningStep(response=error_msg))

                latest_react_messages = react_chat_formatter.format(
                    tools,
                    chat_history=llm_input,
                    current_reasoning=current_reasoning,
                )
                for msg in latest_react_messages:
                    await memory.aput(msg)

                return AgentOutput(
                    response=error_msg,
                    tool_outputs=all_tool_outputs,
                    raw_response=r.raw,
                )

            # If response step, we're done
            if reasoning_step.is_done:
                latest_react_messages = react_chat_formatter.format(
                    tools,
                    chat_history=llm_input,
                    current_reasoning=current_reasoning,
                )
                for msg in latest_react_messages:
                    await memory.aput(msg)

                return AgentOutput(
                    response=reasoning_step.response,
                    tool_outputs=all_tool_outputs,
                    raw_response=r.raw,
                )

    async def _call_llm(
        self,
        ctx: Context,
        llm: LLM,
        llm_input: List[ChatMessage],
        tools: List[AsyncBaseTool],
        mode: AgentMode,
    ) -> AgentOutput:
        """Call the LLM with the given input and tools."""
        if mode == AgentMode.DEFAULT:
            if llm.metadata.is_function_calling_model:
                return await self._call_function_calling_agent(
                    ctx, llm, llm_input, tools
                )
            else:
                return await self._call_react_agent(ctx, llm, llm_input, tools)
        elif mode == AgentMode.REACT:
            return await self._call_react_agent(ctx, llm, llm_input, tools)
        elif mode == AgentMode.FUNCTION_CALLING:
            return await self._call_function_calling_agent(ctx, llm, llm_input, tools)
        else:
            raise ValueError(f"Invalid agent mode: {mode}")

    @step
    async def init_run(self, ctx: Context, ev: StartEvent | AgentOutput) -> AgentInput:
        """Sets up the workflow and validates inputs."""
        if isinstance(ev, StartEvent):
            await self._init_context(ctx)

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
                current_state = await ctx.get("current_state")
                if self.state_prompt and current_state:
                    user_msg.content = self.state_prompt.format(
                        state=current_state, msg=user_msg.content
                    )

                await memory.aput(user_msg)
            else:
                memory.set(chat_history)
                input_messages = memory.get()
        else:
            user_msg_str = await ctx.get("user_msg_str")
            memory: BaseMemory = await ctx.get("memory")
            input_messages = memory.get(input=user_msg_str)

        # send to the current agent
        current_agent = await ctx.get("current_agent")
        return AgentInput(input=input_messages, current_agent=current_agent)

    @step
    async def setup_agent(self, ctx: Context, ev: AgentInput) -> AgentSetup:
        """Main agent handling logic."""
        agent_config: AgentConfig = (await ctx.get("agent_configs"))[ev.current_agent]
        llm_input = ev.input

        # Setup the tools
        tools = list(agent_config.tools or [])
        if agent_config.tool_retriever:
            retrieved_tools = await agent_config.tool_retriever.aretrieve(
                llm_input[-1].content or str(llm_input)
            )
            tools.extend(retrieved_tools)

        if agent_config.can_handoff_to:
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

        return AgentSetup(input=llm_input, current_agent=ev.current_agent, tools=tools)

    @step
    async def run_agent(self, ctx: Context, ev: AgentSetup) -> AgentOutput | StopEvent:
        """Run the agent."""
        current_agent = ev.current_agent
        agent_config: AgentConfig = (await ctx.get("agent_configs"))[current_agent]
        llm = agent_config.llm or Settings.llm

        agent_output: AgentOutput = await self._call_llm(
            ctx, llm, ev.input, ev.tools, agent_config.mode
        )
        ctx.write_event_to_stream(agent_output)
        if agent_output.tool_outputs:
            ctx.write_event_to_stream(agent_output)
            return agent_output
        else:
            return StopEvent(result=agent_output.response)
