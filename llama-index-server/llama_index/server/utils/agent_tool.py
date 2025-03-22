import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Callable, Optional

from pydantic import BaseModel, ConfigDict

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import (
    BaseTool,
    FunctionTool,
    ToolOutput,
    ToolSelection,
)
from llama_index.core.workflow import Context
from llama_index.server.api.models import AgentRunEvent, AgentRunEventType

logger = logging.getLogger("uvicorn")


class ContextAwareTool(FunctionTool, ABC):
    @abstractmethod
    async def acall(self, ctx: Context, input: Any) -> ToolOutput:  # type: ignore
        pass


class ChatWithToolsResponse(BaseModel):
    """
    A tool call response from chat_with_tools.
    """

    tool_calls: Optional[list[ToolSelection]]
    tool_call_message: Optional[ChatMessage]
    generator: Optional[AsyncGenerator[ChatResponse | None, None]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def is_calling_different_tools(self) -> bool:
        tool_names = {tool_call.tool_name for tool_call in self.tool_calls}
        return len(tool_names) > 1

    def has_tool_calls(self) -> bool:
        return self.tool_calls is not None and len(self.tool_calls) > 0

    def tool_name(self) -> str:
        assert self.has_tool_calls()
        assert not self.is_calling_different_tools()
        return self.tool_calls[0].tool_name

    async def full_response(self) -> str:
        assert self.generator is not None
        full_response = ""
        async for chunk in self.generator:
            content = chunk.delta
            if content:
                full_response += content
        return full_response


async def chat_with_tools(  # type: ignore
    llm: FunctionCallingLLM,
    tools: list[BaseTool],
    chat_history: list[ChatMessage],
) -> ChatWithToolsResponse:
    """
    Request LLM to call tools or not.
    This function doesn't change the memory.
    """
    generator = _tool_call_generator(llm, tools, chat_history)
    is_tool_call = await generator.__anext__()
    if is_tool_call:
        # Last chunk is the full response
        # Wait for the last chunk
        full_response = None
        async for chunk in generator:
            full_response = chunk
        assert isinstance(full_response, ChatResponse)
        return ChatWithToolsResponse(
            tool_calls=llm.get_tool_calls_from_response(full_response),
            tool_call_message=full_response.message,
            generator=None,
        )
    else:
        return ChatWithToolsResponse(
            tool_calls=None,
            tool_call_message=None,
            generator=generator,
        )


async def call_tools(
    ctx: Context,
    agent_name: str,
    tools: list[BaseTool],
    tool_calls: list[ToolSelection],
    emit_agent_events: bool = True,
) -> list[ChatMessage]:
    if len(tool_calls) == 0:
        return []

    tools_by_name = {tool.metadata.get_name(): tool for tool in tools}
    if len(tool_calls) == 1:
        return [
            await call_tool(
                ctx,
                tools_by_name[tool_calls[0].tool_name],
                tool_calls[0],
                lambda msg: ctx.write_event_to_stream(
                    AgentRunEvent(
                        name=agent_name,
                        msg=msg,
                    )
                ),
            )
        ]
    # Multiple tool calls, show progress
    tool_msgs: list[ChatMessage] = []

    progress_id = str(uuid.uuid4())
    total_steps = len(tool_calls)
    if emit_agent_events:
        ctx.write_event_to_stream(
            AgentRunEvent(
                name=agent_name,
                msg=f"Making {total_steps} tool calls",
            )
        )
    for i, tool_call in enumerate(tool_calls):
        tool = tools_by_name.get(tool_call.tool_name)
        if not tool:
            tool_msgs.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"Tool {tool_call.tool_name} does not exist",
                )
            )
            continue
        tool_msg = await call_tool(
            ctx,
            tool,
            tool_call,
            event_emitter=lambda msg: ctx.write_event_to_stream(
                AgentRunEvent(
                    name=agent_name,
                    msg=msg,
                    event_type=AgentRunEventType.PROGRESS,
                    data={
                        "id": progress_id,
                        "total": total_steps,
                        "current": i,
                    },
                )
            ),
        )
        tool_msgs.append(tool_msg)
    return tool_msgs


async def call_tool(
    ctx: Context,
    tool: BaseTool,
    tool_call: ToolSelection,
    event_emitter: Optional[Callable[[str], None]],
) -> ChatMessage:
    if event_emitter:
        event_emitter(f"Calling tool {tool_call.tool_name}, {tool_call.tool_kwargs!s}")
    try:
        if isinstance(tool, ContextAwareTool):
            if ctx is None:
                raise ValueError("Context is required for context aware tool")
            # inject context for calling an context aware tool
            response = await tool.acall(ctx=ctx, **tool_call.tool_kwargs)
        else:
            response = await tool.acall(**tool_call.tool_kwargs)  # type: ignore
        return ChatMessage(
            role=MessageRole.TOOL,
            content=str(response.raw_output),
            additional_kwargs={
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            },
        )
    except Exception as e:
        logger.error(f"Got error in tool {tool_call.tool_name}: {e!s}")
        if event_emitter:
            event_emitter(f"Got error in tool {tool_call.tool_name}: {e!s}")
        return ChatMessage(
            role=MessageRole.TOOL,
            content=f"Error: {e!s}",
            additional_kwargs={
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            },
        )


async def _tool_call_generator(
    llm: FunctionCallingLLM,
    tools: list[BaseTool],
    chat_history: list[ChatMessage],
) -> AsyncGenerator[ChatResponse | bool, None]:
    response_stream = await llm.astream_chat_with_tools(
        tools,
        chat_history=chat_history,
        allow_parallel_tool_calls=False,
    )

    full_response = None
    yielded_indicator = False
    async for chunk in response_stream:
        if "tool_calls" not in chunk.message.additional_kwargs:
            # Yield a boolean to indicate whether the response is a tool call
            if not yielded_indicator:
                yield False
                yielded_indicator = True

            # if not a tool call, yield the chunks!
            yield chunk  # type: ignore
        elif not yielded_indicator:
            # Yield the indicator for a tool call
            yield True
            yielded_indicator = True

        full_response = chunk

    if full_response:
        yield full_response  # type: ignore
