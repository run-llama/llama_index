import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from ag_ui.core import RunAgentInput

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, ChatResponse, TextBlock
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import BaseTool, FunctionTool, ToolOutput
from llama_index.core.workflow import Context, Workflow, step
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.protocols.ag_ui.events import (
    MessagesSnapshotWorkflowEvent,
    TextMessageChunkWorkflowEvent,
    ToolCallChunkWorkflowEvent,
)
from llama_index.protocols.ag_ui.utils import (
    llama_index_message_to_ag_ui_message,
    ag_ui_message_to_llama_index_message,
    timestamp,
    validate_tool,
)

DEFAULT_STATE_PROMPT = """<state>
{state}
</state>

{user_input}
"""


class InputEvent(StartEvent):
    input_data: RunAgentInput


class LoopEvent(Event):
    messages: List[ChatMessage]


class ToolCallEvent(Event):
    tool_call_id: str
    tool_name: str
    tool_kwargs: Dict[str, Any]


class ToolCallResultEvent(Event):
    tool_call_id: str
    tool_name: str
    tool_kwargs: Dict[str, Any]
    tool_output: ToolOutput


class AGUIChatWorkflow(Workflow):
    def __init__(
        self,
        llm: Optional[FunctionCallingLLM] = None,
        backend_tools: Optional[List[Union[BaseTool, Callable]]] = None,
        frontend_tools: Optional[List[Union[BaseTool, Callable]]] = None,
        system_prompt: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        **workflow_kwargs: Any,
    ):
        super().__init__(**workflow_kwargs)

        self.llm = llm or Settings.llm
        assert (
            isinstance(self.llm, FunctionCallingLLM)
            and self.llm.metadata.is_function_calling_model
        ), "llm must be a function calling model"

        validated_frontend_tools: List[BaseTool] = [
            validate_tool(tool) for tool in frontend_tools or []
        ]
        validated_backend_tools: List[BaseTool] = [
            validate_tool(tool) for tool in backend_tools or []
        ]

        self.frontend_tools = {
            tool.metadata.name: tool for tool in validated_frontend_tools
        }
        self.backend_tools = {
            tool.metadata.name: tool for tool in validated_backend_tools
        }
        self.initial_state = initial_state or {}
        self.system_prompt = system_prompt

    def _snapshot_messages(self, ctx: Context, chat_history: List[ChatMessage]) -> None:
        # inject tool calls into the assistant message
        for msg in chat_history:
            if msg.role == "assistant":
                tool_calls = self.llm.get_tool_calls_from_response(
                    ChatResponse(message=msg), error_on_no_tool_call=False
                )
                if tool_calls:
                    msg.additional_kwargs["ag_ui_tool_calls"] = [
                        {
                            "id": tool_call.tool_id,
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(tool_call.tool_kwargs),
                        }
                        for tool_call in tool_calls
                    ]

        ag_ui_messages = [llama_index_message_to_ag_ui_message(m) for m in chat_history]

        ctx.write_event_to_stream(
            MessagesSnapshotWorkflowEvent(
                timestamp=timestamp(),
                messages=ag_ui_messages,
            )
        )

    @step
    async def chat(
        self, ctx: Context, ev: InputEvent | LoopEvent
    ) -> Optional[Union[StopEvent, ToolCallEvent]]:
        if isinstance(ev, InputEvent):
            ag_ui_messages = ev.input_data.messages
            chat_history = [
                ag_ui_message_to_llama_index_message(m) for m in ag_ui_messages
            ]

            # State sometimes has unused messages, so we need to remove them
            state = ev.input_data.state
            if isinstance(state, dict):
                state.pop("messages", None)
            elif isinstance(state, str):
                state = json.loads(state)
                state.pop("messages", None)
            else:
                # initial state is not provided, use the default state
                state = self.initial_state.copy()

            # Save state to context for tools to use
            await ctx.set("state", state)

            if state:
                for msg in chat_history[::-1]:
                    if msg.role.value == "user":
                        msg.content = DEFAULT_STATE_PROMPT.format(
                            state=str(state), user_input=msg.content
                        )
                        break

            if self.system_prompt:
                if chat_history[0].role.value == "system":
                    chat_history[0].blocks.append(TextBlock(text=self.system_prompt))
                else:
                    chat_history.insert(
                        0, ChatMessage(role="system", content=self.system_prompt)
                    )

            await ctx.set("chat_history", chat_history)
        else:
            chat_history = await ctx.get("chat_history")

        tools = list(self.frontend_tools.values())
        tools.extend(list(self.backend_tools.values()))

        resp_gen = await self.llm.astream_chat_with_tools(
            tools=tools,
            chat_history=chat_history,
            allow_parallel_tool_calls=True,
        )

        resp_id = str(uuid.uuid4())
        resp = ChatResponse(message=ChatMessage(role="assistant", content=""))

        async for resp in resp_gen:
            if resp.delta:
                ctx.write_event_to_stream(
                    TextMessageChunkWorkflowEvent(
                        role="assistant",
                        delta=resp.delta,
                        timestamp=timestamp(),
                        message_id=resp_id,
                    )
                )

        chat_history.append(resp.message)
        self._snapshot_messages(ctx, [*chat_history])
        await ctx.set("chat_history", chat_history)

        tool_calls = self.llm.get_tool_calls_from_response(
            resp, error_on_no_tool_call=False
        )
        if tool_calls:
            await ctx.set("num_tool_calls", len(tool_calls))
            frontend_tool_calls = [
                tool_call
                for tool_call in tool_calls
                if tool_call.tool_name in self.frontend_tools
            ]
            backend_tool_calls = [
                tool_call
                for tool_call in tool_calls
                if tool_call.tool_name in self.backend_tools
            ]

            # Call backend tools first so that the frontend can return results for frontend tools
            for tool_call in backend_tool_calls:
                ctx.send_event(
                    ToolCallEvent(
                        tool_call_id=tool_call.tool_id,
                        tool_name=tool_call.tool_name,
                        tool_kwargs=tool_call.tool_kwargs,
                    )
                )

                ctx.write_event_to_stream(
                    ToolCallChunkWorkflowEvent(
                        tool_call_id=tool_call.tool_id,
                        tool_call_name=tool_call.tool_name,
                        delta=json.dumps(tool_call.tool_kwargs),
                    )
                )

            for tool_call in frontend_tool_calls:
                ctx.send_event(
                    ToolCallEvent(
                        tool_call_id=tool_call.tool_id,
                        tool_name=tool_call.tool_name,
                        tool_kwargs=tool_call.tool_kwargs,
                    )
                )

                ctx.write_event_to_stream(
                    ToolCallChunkWorkflowEvent(
                        tool_call_id=tool_call.tool_id,
                        tool_call_name=tool_call.tool_name,
                        delta=json.dumps(tool_call.tool_kwargs),
                    )
                )

            return None

        return StopEvent()

    @step
    async def handle_tool_call(
        self, ctx: Context, ev: ToolCallEvent
    ) -> ToolCallResultEvent:
        try:
            all_tools = {**self.frontend_tools, **self.backend_tools}
            tool = all_tools[ev.tool_name]

            kwargs = {**ev.tool_kwargs}
            if isinstance(tool, FunctionTool) and tool.ctx_param_name:
                kwargs[tool.ctx_param_name] = ctx

            tool_output = await tool.acall(**kwargs)
            return ToolCallResultEvent(
                tool_call_id=ev.tool_call_id,
                tool_name=ev.tool_name,
                tool_kwargs=ev.tool_kwargs,
                tool_output=tool_output,
            )
        except Exception as e:
            return ToolCallResultEvent(
                tool_call_id=ev.tool_call_id,
                tool_name=ev.tool_name,
                tool_kwargs=ev.tool_kwargs,
                tool_output=ToolOutput(
                    tool_name=ev.tool_name,
                    content=str(e),
                    raw_input=ev.tool_kwargs,
                    raw_output=str(e),
                    is_error=True,
                ),
            )

    @step
    async def aggregate_tool_calls(
        self, ctx: Context, ev: ToolCallResultEvent
    ) -> Optional[Union[StopEvent, LoopEvent]]:
        num_tool_calls = await ctx.get("num_tool_calls")
        tool_call_results: List[ToolCallResultEvent] = ctx.collect_events(
            ev, [ToolCallResultEvent] * num_tool_calls
        )
        if tool_call_results is None:
            return None

        # organize tool results so that frontend tools are last
        # for backend tools, update the messages snapshot with the tool output
        frontend_tool_calls = [
            tool_result
            for tool_result in tool_call_results
            if tool_result.tool_name in self.frontend_tools
        ]
        backend_tool_calls = [
            tool_result
            for tool_result in tool_call_results
            if tool_result.tool_name in self.backend_tools
        ]

        new_tool_messages = []
        for tool_result in backend_tool_calls:
            new_tool_messages.append(
                ChatMessage(
                    role="tool",
                    content=tool_result.tool_output.content,
                    additional_kwargs={
                        "tool_call_id": tool_result.tool_call_id,
                    },
                )
            )

        # emit a messages snapshot event if there are new messages
        chat_history = await ctx.get("chat_history")
        if new_tool_messages:
            chat_history.extend(new_tool_messages)
            self._snapshot_messages(ctx, [*chat_history])
            await ctx.set("chat_history", chat_history)

        if len(frontend_tool_calls) > 0:
            # Expect frontend tool calls to call back to the agent
            return StopEvent()

        return LoopEvent(messages=chat_history)
