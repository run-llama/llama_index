import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from ag_ui.core import RunAgentInput

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.protocols.ag_ui.events import (
    TextMessageChunkWorkflowEvent,
    ToolCallChunkWorkflowEvent,
)
from llama_index.protocols.ag_ui.utils import (
    ag_ui_message_to_llama_index_message,
    timestamp,
)

DEFAULT_STATE_PROMPT = """<state>
{state}
</state>

{user_input}
"""


class InputEvent(StartEvent):
    input_data: RunAgentInput


class AGUIChatWorkflow(Workflow):
    def __init__(
        self,
        llm: Optional[FunctionCallingLLM] = None,
        tools: Optional[List[Union[BaseTool, Callable]]] = None,
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

        validated_tools: List[BaseTool] = []
        tools = tools or []
        for tool in tools:
            if isinstance(tool, BaseTool):
                validated_tools.append(tool)
            elif callable(tool):
                validated_tools.append(FunctionTool.from_defaults(tool))
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")

        self.tools = validated_tools
        self.initial_state = initial_state or {}
        self.system_prompt = system_prompt

    @step
    async def chat(self, ctx: Context, ev: InputEvent) -> StopEvent:
        ag_ui_messages = ev.input_data.messages
        llama_index_messages = [
            ag_ui_message_to_llama_index_message(m) for m in ag_ui_messages
        ]
        chat_history = llama_index_messages[:-1]
        user_input = llama_index_messages[-1].content

        if self.system_prompt:
            if chat_history[0].role.value == "system":
                chat_history[0].content += "\n\n" + self.system_prompt
            else:
                chat_history.insert(
                    0, ChatMessage(role="system", content=self.system_prompt)
                )

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
            user_input = DEFAULT_STATE_PROMPT.format(
                state=str(state), user_input=user_input
            )

        resp_gen = await self.llm.astream_chat_with_tools(
            tools=self.tools,
            user_msg=user_input,
            chat_history=chat_history,
            allow_parallel_tool_calls=True,
        )

        resp_id = str(uuid.uuid4())
        resp = ChatMessage(role="assistant", content="")

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

        tool_calls = self.llm.get_tool_calls_from_response(
            resp, error_on_no_tool_call=False
        )
        if tool_calls:
            tools_by_name = {t.metadata.name: t for t in self.tools}
            for tool_call in tool_calls:
                tool = tools_by_name[tool_call.tool_name]
                ctx.write_event_to_stream(
                    ToolCallChunkWorkflowEvent(
                        tool_call_id=tool_call.tool_id,
                        tool_call_name=tool_call.tool_name,
                        delta=json.dumps(tool_call.tool_kwargs, sort_keys=True),
                        timestamp=timestamp(),
                    )
                )

                # Run tools with ctx if indicated
                if isinstance(tool, FunctionTool) and tool.ctx_param_name:
                    tool_kwargs = {**tool_call.tool_kwargs, tool.ctx_param_name: ctx}
                    _ = await tool.acall(**tool_kwargs)
                else:
                    _ = await tool.acall(**tool_call.tool_kwargs)

        return StopEvent()
