from abc import ABCMeta
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

from llama_index.core.agent.workflow.base_agent import (
    BaseWorkflowAgent,
    DEFAULT_AGENT_NAME,
    DEFAULT_AGENT_DESCRIPTION,
)
from llama_index.core.agent.workflow.function_agent import FunctionAgent
from llama_index.core.agent.workflow.react_agent import ReActAgent
from llama_index.core.agent.workflow.workflow_events import (
    ToolCall,
    ToolCallResult,
    AgentInput,
    AgentSetup,
    AgentOutput,
)
from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptMixin, PromptMixinType, PromptDictType
from llama_index.core.tools import (
    BaseTool,
    AsyncBaseTool,
    FunctionTool,
    ToolOutput,
    ToolSelection,
    adapt_to_async_tool,
)
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.core.workflow.checkpointer import CheckpointCallback
from llama_index.core.workflow.handler import WorkflowHandler
from llama_index.core.workflow.workflow import WorkflowMeta
from llama_index.core.settings import Settings


DEFAULT_HANDOFF_PROMPT = """Useful for handing off to another agent.
If you are currently not equipped to handle the user's request, or another agent is better suited to handle the request, please hand off to the appropriate agent.

Currently available agents:
{agent_info}
"""

DEFAULT_STATE_PROMPT = """Current state:
{state}

Current message:
{msg}
"""

DEFAULT_HANDOFF_OUTPUT_PROMPT = "Agent {to_agent} is now handling the request due to the following reason: {reason}.\nPlease continue with the current request."


async def handoff(ctx: Context, to_agent: str, reason: str) -> str:
    """Handoff control of that chat to the given agent."""
    agents: list[str] = await ctx.get("agents")
    current_agent_name: str = await ctx.get("current_agent_name")
    if to_agent not in agents:
        valid_agents = ", ".join([x for x in agents if x != current_agent_name])
        return f"Agent {to_agent} not found. Please select a valid agent to hand off to. Valid agents: {valid_agents}"

    await ctx.set("next_agent", to_agent)
    handoff_output_prompt = await ctx.get(
        "handoff_output_prompt", default=DEFAULT_HANDOFF_OUTPUT_PROMPT
    )

    return handoff_output_prompt.format(to_agent=to_agent, reason=reason)


class AgentWorkflowMeta(WorkflowMeta, ABCMeta):
    """Metaclass for AgentWorkflow that inherits from WorkflowMeta."""


class AgentWorkflow(Workflow, PromptMixin, metaclass=AgentWorkflowMeta):
    """A workflow for managing multiple agents with handoffs."""

    def __init__(
        self,
        agents: List[BaseWorkflowAgent],
        initial_state: Optional[Dict] = None,
        root_agent: Optional[str] = None,
        handoff_prompt: Optional[Union[str, BasePromptTemplate]] = None,
        handoff_output_prompt: Optional[Union[str, BasePromptTemplate]] = None,
        state_prompt: Optional[Union[str, BasePromptTemplate]] = None,
        timeout: Optional[float] = None,
        **workflow_kwargs: Any,
    ):
        super().__init__(timeout=timeout, **workflow_kwargs)
        if not agents:
            raise ValueError("At least one agent must be provided")

        # Raise an error if any agent has no name or no description
        if len(agents) > 1 and any(
            agent.name == DEFAULT_AGENT_NAME for agent in agents
        ):
            raise ValueError("All agents must have a name in a multi-agent workflow")

        if len(agents) > 1 and any(
            agent.description == DEFAULT_AGENT_DESCRIPTION for agent in agents
        ):
            raise ValueError(
                "All agents must have a description in a multi-agent workflow"
            )

        self.agents = {cfg.name: cfg for cfg in agents}
        if len(agents) == 1:
            root_agent = agents[0].name
        elif root_agent is None:
            raise ValueError("Exactly one root agent must be provided")
        else:
            root_agent = root_agent

        if root_agent not in self.agents:
            raise ValueError(f"Root agent {root_agent} not found in provided agents")

        self.root_agent = root_agent
        self.initial_state = initial_state or {}

        handoff_prompt = handoff_prompt or DEFAULT_HANDOFF_PROMPT
        if isinstance(handoff_prompt, str):
            handoff_prompt = PromptTemplate(handoff_prompt)
            if "{agent_info}" not in handoff_prompt.get_template():
                raise ValueError("Handoff prompt must contain {agent_info}")
        self.handoff_prompt = handoff_prompt

        handoff_output_prompt = handoff_output_prompt or DEFAULT_HANDOFF_OUTPUT_PROMPT
        if isinstance(handoff_output_prompt, str):
            handoff_output_prompt = PromptTemplate(handoff_output_prompt)
            if (
                "{to_agent}" not in handoff_output_prompt.get_template()
                or "{reason}" not in handoff_output_prompt.get_template()
            ):
                raise ValueError(
                    "Handoff output prompt must contain {to_agent} and {reason}"
                )
        self.handoff_output_prompt = handoff_output_prompt

        state_prompt = state_prompt or DEFAULT_STATE_PROMPT
        if isinstance(state_prompt, str):
            state_prompt = PromptTemplate(state_prompt)
            if (
                "{state}" not in state_prompt.get_template()
                or "{msg}" not in state_prompt.get_template()
            ):
                raise ValueError("State prompt must contain {state} and {msg}")
        self.state_prompt = state_prompt

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "handoff_prompt": self.handoff_prompt,
            "handoff_output_prompt": self.handoff_output_prompt,
            "state_prompt": self.state_prompt,
        }

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {agent.name: agent for agent in self.agents.values()}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        """Update prompts."""
        if "handoff_prompt" in prompts_dict:
            self.handoff_prompt = prompts_dict["handoff_prompt"]
        if "handoff_output_prompt" in prompts_dict:
            self.handoff_output_prompt = prompts_dict["handoff_output_prompt"]
        if "state_prompt" in prompts_dict:
            self.state_prompt = prompts_dict["state_prompt"]

    def _ensure_tools_are_async(
        self, tools: Sequence[BaseTool]
    ) -> Sequence[AsyncBaseTool]:
        """Ensure all tools are async."""
        return [adapt_to_async_tool(tool) for tool in tools]

    def _get_handoff_tool(
        self, current_agent: BaseWorkflowAgent
    ) -> Optional[AsyncBaseTool]:
        """Creates a handoff tool for the given agent."""
        # Do not create a handoff tool if there is only one agent
        if len(self.agents) == 1:
            return None

        agent_info = {cfg.name: cfg.description for cfg in self.agents.values()}

        # Filter out agents that the current agent cannot handoff to
        configs_to_remove = []
        for name in agent_info:
            if name == current_agent.name:
                configs_to_remove.append(name)
            elif (
                current_agent.can_handoff_to is not None
                and name not in current_agent.can_handoff_to
            ):
                configs_to_remove.append(name)

        for name in configs_to_remove:
            agent_info.pop(name)

        if not agent_info:
            return None

        fn_tool_prompt = self.handoff_prompt.format(agent_info=str(agent_info))
        return FunctionTool.from_defaults(
            async_fn=handoff, description=fn_tool_prompt, return_direct=True
        )

    async def get_tools(
        self, agent_name: str, input_str: Optional[str] = None
    ) -> Sequence[AsyncBaseTool]:
        """Get tools for the given agent."""
        agent_tools = self.agents[agent_name].tools or []
        tools = [*agent_tools]
        retriever = self.agents[agent_name].tool_retriever
        if retriever is not None:
            retrieved_tools = await retriever.aretrieve(input_str or "")
            tools.extend(retrieved_tools)

        if (
            self.agents[agent_name].can_handoff_to
            or self.agents[agent_name].can_handoff_to is None
        ):
            handoff_tool = self._get_handoff_tool(self.agents[agent_name])
            if handoff_tool:
                tools.append(handoff_tool)

        return self._ensure_tools_are_async(cast(List[BaseTool], tools))

    async def _init_context(self, ctx: Context, ev: StartEvent) -> None:
        """Initialize the context once, if needed."""
        if not await ctx.get("memory", default=None):
            default_memory = ev.get("memory", default=None)
            default_memory = default_memory or ChatMemoryBuffer.from_defaults(
                llm=self.agents[self.root_agent].llm or Settings.llm
            )
            await ctx.set("memory", default_memory)
        if not await ctx.get("agents", default=None):
            await ctx.set("agents", list(self.agents.keys()))
        if not await ctx.get("state", default=None):
            await ctx.set("state", self.initial_state)
        if not await ctx.get("current_agent_name", default=None):
            await ctx.set("current_agent_name", self.root_agent)
        if not await ctx.get("handoff_output_prompt", default=None):
            await ctx.set(
                "handoff_output_prompt", self.handoff_output_prompt.get_template()
            )

    async def _call_tool(
        self,
        ctx: Context,
        tool: AsyncBaseTool,
        tool_input: dict,
    ) -> ToolOutput:
        """Call the given tool with the given input."""
        try:
            if isinstance(tool, FunctionTool) and tool.requires_context:
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

    @step
    async def init_run(self, ctx: Context, ev: StartEvent) -> AgentInput:
        """Sets up the workflow and validates inputs."""
        await self._init_context(ctx, ev)

        user_msg: Optional[Union[str, ChatMessage]] = ev.get("user_msg")
        chat_history: Optional[List[ChatMessage]] = ev.get("chat_history", [])

        # Convert string user_msg to ChatMessage
        if isinstance(user_msg, str):
            user_msg = ChatMessage(role="user", content=user_msg)

        # Add messages to memory
        memory: BaseMemory = await ctx.get("memory")

        # First set chat history if it exists
        if chat_history:
            memory.set(chat_history)

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
            await ctx.set("user_msg_str", content_str)
        elif chat_history:
            # If no user message, use the last message from chat history as user_msg_str
            content_str = "\n".join(
                [
                    block.text
                    for block in chat_history[-1].blocks
                    if isinstance(block, TextBlock)
                ]
            )
            await ctx.set("user_msg_str", content_str)
        else:
            raise ValueError("Must provide either user_msg or chat_history")

        # Get all messages from memory
        input_messages = memory.get()

        # send to the current agent
        current_agent_name: str = await ctx.get("current_agent_name")
        return AgentInput(input=input_messages, current_agent_name=current_agent_name)

    @step
    async def setup_agent(self, ctx: Context, ev: AgentInput) -> AgentSetup:
        """Main agent handling logic."""
        current_agent_name = ev.current_agent_name
        agent = self.agents[current_agent_name]
        llm_input = ev.input

        if agent.system_prompt:
            llm_input = [
                ChatMessage(role="system", content=agent.system_prompt),
                *llm_input,
            ]

        state = await ctx.get("state", default=None)
        if state:
            # update last message with current state
            for block in llm_input[-1].blocks[::-1]:
                if isinstance(block, TextBlock):
                    block.text = self.state_prompt.format(state=state, msg=block.text)
                    break

        return AgentSetup(
            input=llm_input,
            current_agent_name=ev.current_agent_name,
        )

    @step
    async def run_agent_step(self, ctx: Context, ev: AgentSetup) -> AgentOutput:
        """Run the agent."""
        memory: BaseMemory = await ctx.get("memory")
        agent = self.agents[ev.current_agent_name]
        user_msg_str = await ctx.get("user_msg_str")
        tools = await self.get_tools(ev.current_agent_name, user_msg_str or "")

        agent_output = await agent.take_step(
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
    ) -> Union[StopEvent, ToolCall, None]:
        if not ev.tool_calls:
            agent = self.agents[ev.current_agent_name]
            memory: BaseMemory = await ctx.get("memory")
            output = await agent.finalize(ctx, ev, memory)

            cur_tool_calls: List[ToolCallResult] = await ctx.get(
                "current_tool_calls", default=[]
            )
            output.tool_calls.extend(cur_tool_calls)  # type: ignore
            await ctx.set("current_tool_calls", [])

            return StopEvent(result=output)

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
        ctx.write_event_to_stream(
            ToolCall(
                tool_name=ev.tool_name,
                tool_kwargs=ev.tool_kwargs,
                tool_id=ev.tool_id,
            )
        )

        current_agent_name = await ctx.get("current_agent_name")
        tools = await self.get_tools(current_agent_name, ev.tool_name)
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
        num_tool_calls = await ctx.get("num_tool_calls", default=0)
        if num_tool_calls == 0:
            raise ValueError("No tool calls found, cannot aggregate results.")

        tool_call_results: list[ToolCallResult] = ctx.collect_events(  # type: ignore
            ev, expected=[ToolCallResult] * num_tool_calls
        )
        if not tool_call_results:
            return None

        memory: BaseMemory = await ctx.get("memory")
        agent_name: str = await ctx.get("current_agent_name")
        agent: BaseWorkflowAgent = self.agents[agent_name]

        # track tool calls made during a .run() call
        cur_tool_calls: List[ToolCallResult] = await ctx.get(
            "current_tool_calls", default=[]
        )
        cur_tool_calls.extend(tool_call_results)
        await ctx.set("current_tool_calls", cur_tool_calls)

        await agent.handle_tool_call_results(ctx, tool_call_results, memory)

        # set the next agent, if needed
        # the handoff tool sets this
        next_agent_name = await ctx.get("next_agent", default=None)
        if next_agent_name:
            await ctx.set("current_agent_name", next_agent_name)

        if any(
            tool_call_result.return_direct for tool_call_result in tool_call_results
        ):
            # if any tool calls return directly, take the first one
            return_direct_tool = next(
                tool_call_result
                for tool_call_result in tool_call_results
                if tool_call_result.return_direct
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
                current_agent_name=agent.name,
            )
            result = await agent.finalize(ctx, result, memory)

            # we don't want to stop the system if we're just handing off
            if return_direct_tool.tool_name != "handoff":
                await ctx.set("current_tool_calls", [])
                return StopEvent(result=result)

        user_msg_str = await ctx.get("user_msg_str")
        input_messages = memory.get(input=user_msg_str)

        # get this again, in case it changed
        agent_name = await ctx.get("current_agent_name")
        agent = self.agents[agent_name]

        return AgentInput(input=input_messages, current_agent_name=agent.name)

    def run(
        self,
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        ctx: Optional[Context] = None,
        stepwise: bool = False,
        checkpoint_callback: Optional[CheckpointCallback] = None,
        **kwargs: Any,
    ) -> WorkflowHandler:
        return super().run(
            user_msg=user_msg,
            chat_history=chat_history,
            memory=memory,
            ctx=ctx,
            stepwise=stepwise,
            checkpoint_callback=checkpoint_callback,
            **kwargs,
        )

    @classmethod
    def from_tools_or_functions(
        cls,
        tools_or_functions: List[Union[BaseTool, Callable]],
        llm: Optional[LLM] = None,
        system_prompt: Optional[str] = None,
        state_prompt: Optional[Union[str, BasePromptTemplate]] = None,
        initial_state: Optional[dict] = None,
        timeout: Optional[float] = None,
        verbose: bool = False,
    ) -> "AgentWorkflow":
        """Initializes an AgentWorkflow from a list of tools or functions.

        The workflow will be initialized with a single agent that uses the provided tools or functions.

        If the LLM is a function calling model, the workflow will use the FunctionAgent.
        Otherwise, it will use the ReActAgent.
        """
        llm = llm or Settings.llm
        agent_cls = (
            FunctionAgent if llm.metadata.is_function_calling_model else ReActAgent
        )

        tools = [
            FunctionTool.from_defaults(fn=tool)
            if not isinstance(tool, BaseTool)
            else tool
            for tool in tools_or_functions
        ]
        return cls(
            agents=[
                agent_cls(
                    name="Agent",
                    description="A single agent that uses the provided tools or functions.",
                    tools=tools,
                    llm=llm,
                    system_prompt=system_prompt,
                )
            ],
            state_prompt=state_prompt,
            initial_state=initial_state,
            timeout=timeout,
            verbose=verbose,
        )
