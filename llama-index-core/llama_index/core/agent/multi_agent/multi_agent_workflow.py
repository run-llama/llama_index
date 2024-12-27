import uuid
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
    BaseReasoningStep,
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


async def handoff(ctx: Context, to_agent: str, reason: str) -> HandoffEvent:
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
        if not await ctx.get("current_state", default=None):
            await ctx.set("current_state", self.initial_state)
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
        current_agent = await ctx.get("current_agent")
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
                    current_agent=current_agent,
                )
            )

        current_llm_input.append(r.message)
        await memory.aput(r.message)
        tool_calls = llm.get_tool_calls_from_response(r, error_on_no_tool_call=False)

        return AgentOutput(
            response=r.message.content,
            tool_calls=tool_calls or [],
            raw_response=r.raw,
            current_agent=current_agent,
        )

        # all_tool_results = []
        # while tool_calls:
        #     tool_results: List[ToolOutput] = []
        #     tool_ids: List[str] = []
        #     jobs = []
        #     should_return_direct = False
        #     for tool_call in tool_calls:
        #         tool_ids.append(tool_call.tool_id)
        #         if tool_call.tool_name not in tools_by_name:
        #             tool_results.append(
        #                 ToolOutput(
        #                     content=f"Tool {tool_call.tool_name} not found. Please select a tool that is available.",
        #                     tool_name=tool_call.tool_name,
        #                     raw_input=tool_call.tool_kwargs,
        #                     raw_output=None,
        #                     is_error=True,
        #                 )
        #             )
        #         else:
        #             tool = tools_by_name[tool_call.tool_name]
        #             if tool.metadata.return_direct:
        #                 should_return_direct = True

        #             jobs.append(
        #                 self._call_tool(
        #                     ctx,
        #                     tool,
        #                     tool_call.tool_kwargs,
        #                 )
        #             )

        #     tool_results.extend(await asyncio.gather(*jobs))
        #     all_tool_results.extend(tool_results)
        #     tool_messages = [
        #         ChatMessage(
        #             role="tool",
        #             content=str(result),
        #             additional_kwargs={"tool_call_id": tool_id},
        #         )
        #         for result, tool_id in zip(tool_results, tool_ids)
        #     ]

        #     for tool_message in tool_messages:
        #         await memory.aput(tool_message)

        #     if should_return_direct:
        #         return AgentOutput(
        #             response=tool_results[0].content,
        #             tool_outputs=all_tool_results,
        #             raw_response=None,
        #             current_agent=current_agent,
        #         )

        #     current_llm_input.extend(tool_messages)
        #     response = await llm.astream_chat_with_tools(
        #         tools, chat_history=current_llm_input, allow_parallel_tool_calls=True
        #     )
        #     async for r in response:
        #         tool_calls = llm.get_tool_calls_from_response(
        #             r, error_on_no_tool_call=False
        #         )
        #         ctx.write_event_to_stream(
        #             AgentStream(
        #                 delta=r.delta or "",
        #                 tool_calls=tool_calls or [],
        #                 raw_response=r.raw,
        #                 current_agent=current_agent,
        #             )
        #         )

        #     current_llm_input.append(r.message)
        #     await memory.aput(r.message)
        #     tool_calls = llm.get_tool_calls_from_response(
        #         r, error_on_no_tool_call=False
        #     )

        # return AgentOutput(
        #     response=r.message.content,
        #     tool_outputs=all_tool_results,
        #     raw_response=r.raw,
        #     current_agent=current_agent,
        # )

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

            return AgentOutput(
                response=r.message.content,
                tool_calls=[],
                raw_response=r.raw,
                current_agent=current_agent,
            )

        current_reasoning.append(reasoning_step)
        await ctx.set("current_reasoning", current_reasoning)

        # If response step, we're done
        if reasoning_step.is_done:
            reasoning_str = "\n".join([x.get_content() for x in current_reasoning])
            reasoning_msg = ChatMessage(role="assistant", content=reasoning_str)
            await memory.aput(reasoning_msg)

            response = (
                reasoning_step.response
                if hasattr(reasoning_step, "response")
                else reasoning_step.get_content()
            )
            return AgentOutput(
                response=response,
                tool_calls=[],
                raw_response=r.raw,
                current_agent=current_agent,
            )

        reasoning_step = cast(ActionReasoningStep, reasoning_step)
        if not isinstance(reasoning_step, ActionReasoningStep):
            raise ValueError(f"Expected ActionReasoningStep, got {reasoning_step}")

        # Call tool
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

        # tools_by_name = {tool.metadata.name: tool for tool in tools}
        # tool = None
        # if reasoning_step.action not in tools_by_name:
        #     tool_output = ToolOutput(
        #         content=f"Error: No such tool named `{reasoning_step.action}`.",
        #         tool_name=reasoning_step.action,
        #         raw_input={"kwargs": reasoning_step.action_input},
        #         raw_output=None,
        #         is_error=True,
        #     )
        # else:
        #     tool = tools_by_name[reasoning_step.action]
        #     tool_output = await self._call_tool(
        #         ctx,
        #         tool,
        #         reasoning_step.action_input,
        #     )
        #     all_tool_outputs.append(tool_output)

        # # Add observation to chat history
        # current_reasoning.append(
        #     ObservationReasoningStep(
        #         observation=str(tool_output),
        #         return_direct=tool.metadata.return_direct,
        #     )
        # )

        # if tool and tool.metadata.return_direct:
        #     current_reasoning.append(
        #         ResponseReasoningStep(response=tool_output.content)
        #     )
        #     latest_react_messages = react_chat_formatter.format(
        #         tools,
        #         chat_history=llm_input,
        #         current_reasoning=current_reasoning,
        #     )
        #     for msg in latest_react_messages:
        #         await memory.aput(msg)

        #     return AgentOutput(
        #         response=tool_output.content,
        #         tool_outputs=all_tool_outputs,
        #         raw_response=r.raw,
        #         current_agent=current_agent,
        #     )

        # # Get next action from LLM
        # input_chat = react_chat_formatter.format(
        #     tools,
        #     chat_history=llm_input,
        #     current_reasoning=current_reasoning,
        # )
        # response = await llm.astream_chat(input_chat)
        # async for r in response:
        #     ctx.write_event_to_stream(
        #         AgentStream(
        #             delta=r.delta or "",
        #             tool_calls=[],
        #             current_agent=current_agent,
        #             raw_response=r.raw,
        #         )
        #     )

        # await memory.aput(r.message)

        # # Parse next reasoning step
        # message_content = r.message.content
        # if not message_content:
        #     raise ValueError("Got empty message")

        # try:
        #     reasoning_step = output_parser.parse(
        #         message_content, is_streaming=False
        #     )
        # except ValueError as e:
        #     # If we can't parse the output, return an error message
        #     error_msg = f"Error: Could not parse output. Please follow the thought-action-input format. Try again. Details: {e!s}"

        #     current_reasoning.append(
        #         ObservationReasoningStep(observation=error_msg)
        #     )

        #     latest_react_messages = react_chat_formatter.format(
        #         tools,
        #         chat_history=llm_input,
        #         current_reasoning=current_reasoning,
        #     )
        #     for msg in latest_react_messages:
        #         await memory.aput(msg)

        #     return AgentOutput(
        #         response=error_msg,
        #         tool_outputs=all_tool_outputs,
        #         raw_response=r.raw,
        #         current_agent=current_agent,
        #     )

        # # If response step, we're done
        # if reasoning_step.is_done:
        #     latest_react_messages = react_chat_formatter.format(
        #         tools,
        #         chat_history=llm_input,
        #         current_reasoning=current_reasoning,
        #     )
        #     for msg in latest_react_messages:
        #         await memory.aput(msg)

        #     return AgentOutput(
        #         response=reasoning_step.response,
        #         tool_outputs=all_tool_outputs,
        #         raw_response=r.raw,
        #         current_agent=current_agent,
        #     )

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
        elif mode == AgentMode.FUNCTION:
            return await self._call_function_calling_agent(ctx, llm, llm_input, tools)
        else:
            raise ValueError(f"Invalid agent mode: {mode}")

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
            current_state = await ctx.get("current_state")
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
        agent_config: AgentConfig = (await ctx.get("agent_configs"))[ev.current_agent]
        llm_input = ev.input

        # Setup the tools
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
            ctx, llm, ev.input, ev.tools, agent_config.mode
        )
        ctx.write_event_to_stream(agent_output)

        return agent_output

    @step
    async def parse_agent_output(
        self, ctx: Context, ev: AgentOutput
    ) -> StopEvent | ToolCall:
        pass

    def _add_tool_call_result_to_memory(self, ctx: Context, ev: ToolCallResult) -> None:
        """Either adds to memory or adds to the react reasoning list."""

    @step
    async def call_tool(self, ctx: Context, ev: ToolCall) -> ToolCallResult:
        pass

    @step
    async def aggregate_tool_results(
        self, ctx: Context, ev: ToolCallResult
    ) -> AgentInput | None:
        """Aggregate tool results and return the next agent input."""
        num_tool_calls = await ctx.get("num_tool_calls", default=0)
        if num_tool_calls == 0:
            raise ValueError("No tool calls found, cannot aggregate results.")

        tool_call_results = ctx.collect_events(
            ev, expected=[ToolCallResult] * num_tool_calls
        )
        if not tool_call_results:
            return None

        user_msg_str = await ctx.get("user_msg_str")
        memory: BaseMemory = await ctx.get("memory")
        input_messages = memory.get(input=user_msg_str)
        current_agent = await ctx.get("current_agent")

        return AgentInput(input=input_messages, current_agent=current_agent)
