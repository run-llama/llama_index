from typing import List, Optional, Sequence

from llama_index.core.agent.workflow.agent_context import AgentContext
from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCallResult,
)
from llama_index.core.base.llms.types import ChatResponse, ThinkingBlock
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemory
from llama_index.core.tools import AsyncBaseTool
from llama_index.core.workflow import Context


class FunctionAgent(BaseWorkflowAgent):
    """Function calling agent implementation."""

    scratchpad_key: str = "scratchpad"
    initial_tool_choice: Optional[str] = Field(
        default=None,
        description="The tool to try and force to call on the first iteration of the agent.",
    )
    allow_parallel_tool_calls: bool = Field(
        default=True,
        description="If True, the agent will call multiple tools in parallel. If False, the agent will call tools sequentially.",
    )

    async def _get_response(
        self, current_llm_input: List[ChatMessage], tools: Sequence[AsyncBaseTool]
    ) -> ChatResponse:
        chat_kwargs = {
            "chat_history": current_llm_input,
            "allow_parallel_tool_calls": self.allow_parallel_tool_calls,
            "tools": tools,
        }

        # Only add tool choice if set and if its the first response
        if (
            self.initial_tool_choice is not None
            and current_llm_input[-1].role == "user"
        ):
            chat_kwargs["tool_choice"] = self.initial_tool_choice

        return await self.llm.achat_with_tools(  # type: ignore
            **chat_kwargs
        )

    async def _get_streaming_response(
        self,
        ctx: AgentContext,
        current_llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
    ) -> ChatResponse:
        chat_kwargs = {
            "chat_history": current_llm_input,
            "tools": tools,
            "allow_parallel_tool_calls": self.allow_parallel_tool_calls,
        }

        # Only add tool choice if set and if its the first response
        if (
            self.initial_tool_choice is not None
            and current_llm_input[-1].role == "user"
        ):
            chat_kwargs["tool_choice"] = self.initial_tool_choice

        response = await self.llm.astream_chat_with_tools(  # type: ignore
            **chat_kwargs
        )
        # last_chat_response will be used later, after the loop.
        # We initialize it so it's valid even when 'response' is empty
        last_chat_response = ChatResponse(message=ChatMessage())
        async for last_chat_response in response:
            tool_calls = self.llm.get_tool_calls_from_response(  # type: ignore
                last_chat_response, error_on_no_tool_call=False
            )
            raw = (
                last_chat_response.raw.model_dump()
                if isinstance(last_chat_response.raw, BaseModel)
                else last_chat_response.raw
            )
            ctx.write_event_to_stream(
                AgentStream(
                    delta=last_chat_response.delta or "",
                    response=last_chat_response.message.content or "",
                    tool_calls=tool_calls or [],
                    raw=raw,
                    current_agent_name=self.name,
                    thinking_delta=last_chat_response.additional_kwargs.get(
                        "thinking_delta", None
                    ),
                )
            )

        return last_chat_response

    @staticmethod
    def _extract_reasoning_content(chat_response: ChatResponse) -> Optional[str]:
        """
        Best-effort recovery of a final answer that a model placed in its
        reasoning channel instead of the normal text content.

        Looks, in order, at:
        1. ThinkingBlock content on the message (all non-empty blocks, joined);
        2. a ``reasoning_content`` string in ``message.additional_kwargs``;
        3. a ``reasoning_content`` string in the raw provider payload
           (top level or under an OpenAI-style ``choices[0].message``).

        Returns ``None`` when no reasoning content is available.
        """
        message = chat_response.message

        thinking = [
            block.content
            for block in message.blocks
            if isinstance(block, ThinkingBlock) and block.content
        ]
        if thinking:
            return "".join(thinking)

        kwargs_reasoning = message.additional_kwargs.get("reasoning_content")
        if isinstance(kwargs_reasoning, str) and kwargs_reasoning:
            return kwargs_reasoning

        raw = chat_response.raw
        raw_dict = raw.model_dump() if isinstance(raw, BaseModel) else raw
        if isinstance(raw_dict, dict):
            raw_reasoning = raw_dict.get("reasoning_content")
            if isinstance(raw_reasoning, str) and raw_reasoning:
                return raw_reasoning
            choices = raw_dict.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    inner = first.get("message")
                    if isinstance(inner, dict):
                        inner_reasoning = inner.get("reasoning_content")
                        if isinstance(inner_reasoning, str) and inner_reasoning:
                            return inner_reasoning

        return None

    async def take_step(
        self,
        ctx: AgentContext,
        llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
        memory: BaseMemory,
    ) -> AgentOutput:
        """Take a single step with the function calling agent."""
        if not self.llm.metadata.is_function_calling_model:
            raise ValueError("LLM must be a FunctionCallingLLM")

        scratchpad: List[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )
        current_llm_input = [*llm_input, *scratchpad]

        ctx.write_event_to_stream(
            AgentInput(input=current_llm_input, current_agent_name=self.name)
        )

        if self.streaming:
            last_chat_response = await self._get_streaming_response(
                ctx, current_llm_input, tools
            )
        else:
            last_chat_response = await self._get_response(current_llm_input, tools)

        tool_calls = self.llm.get_tool_calls_from_response(  # type: ignore
            last_chat_response, error_on_no_tool_call=False
        )

        # Some OpenAI-compatible models (e.g. Kimi-K2.5) return the final answer in
        # `reasoning_content` / a ThinkingBlock instead of the normal text content.
        # When the step produced no tool calls and no text content, FunctionAgent would
        # otherwise silently return an empty answer -- unlike ReActAgent, which validates
        # for empty content. Promote the reasoning content to text so the answer is not
        # lost. This never overrides content that is already present.
        if not tool_calls and not last_chat_response.message.content:
            reasoning_content = self._extract_reasoning_content(last_chat_response)
            if reasoning_content:
                last_chat_response = ChatResponse(
                    message=ChatMessage(
                        role=last_chat_response.message.role,
                        content=reasoning_content,
                        additional_kwargs=last_chat_response.message.additional_kwargs,
                    ),
                    delta=reasoning_content,
                    raw=last_chat_response.raw,
                    additional_kwargs=last_chat_response.additional_kwargs,
                )

        # only add to scratchpad if we didn't select the handoff tool
        scratchpad.append(last_chat_response.message)
        await ctx.store.set(self.scratchpad_key, scratchpad)

        raw = (
            last_chat_response.raw.model_dump()
            if isinstance(last_chat_response.raw, BaseModel)
            else last_chat_response.raw
        )
        return AgentOutput(
            response=last_chat_response.message,
            tool_calls=tool_calls or [],
            raw=raw,
            current_agent_name=self.name,
        )

    async def handle_tool_call_results(
        self, ctx: Context, results: List[ToolCallResult], memory: BaseMemory
    ) -> None:
        """Handle tool call results for function calling agent."""
        scratchpad: List[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )

        for tool_call_result in results:
            scratchpad.append(
                ChatMessage(
                    role="tool",
                    blocks=tool_call_result.tool_output.blocks,
                    additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                )
            )

            if (
                tool_call_result.return_direct
                and tool_call_result.tool_name != "handoff"
            ):
                scratchpad.append(
                    ChatMessage(
                        role="assistant",
                        content=str(tool_call_result.tool_output.content),
                        additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                    )
                )
                break

        await ctx.store.set(self.scratchpad_key, scratchpad)

    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        """
        Finalize the function calling agent.

        Adds all in-progress messages to memory.
        """
        scratchpad: List[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )
        await memory.aput_messages(scratchpad)

        # reset scratchpad
        await ctx.store.set(self.scratchpad_key, [])

        return output
