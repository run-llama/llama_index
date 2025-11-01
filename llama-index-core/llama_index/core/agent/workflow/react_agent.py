import uuid
from typing import List, Sequence, Optional, cast

from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCallResult,
)
from llama_index.core.base.llms.types import ChatResponse, TextBlock
from llama_index.core.bridge.pydantic import BaseModel, Field, model_validator
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.memory import BaseMemory
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.tools import AsyncBaseTool
from llama_index.core.workflow import Context


def default_formatter(fields: Optional[dict] = None) -> ReActChatFormatter:
    """Sets up a default formatter so that the proper react header is set."""
    fields = fields or {}
    return ReActChatFormatter.from_defaults(context=fields.get("system_prompt", None))


class ReActAgent(BaseWorkflowAgent):
    """React agent implementation."""

    reasoning_key: str = "current_reasoning"
    output_parser: ReActOutputParser = Field(
        default_factory=ReActOutputParser, description="The react output parser"
    )
    formatter: ReActChatFormatter = Field(
        default_factory=default_formatter,
        description="The react chat formatter to format the reasoning steps and chat history into an llm input.",
    )

    @model_validator(mode="after")
    def validate_formatter(self) -> "ReActAgent":
        """Validate the formatter."""
        if (
            self.formatter.context
            and self.system_prompt
            and self.system_prompt not in self.formatter.context
        ):
            self.formatter.context = (
                self.system_prompt + "\n\n" + self.formatter.context.strip()
            )
        elif not self.formatter.context and self.system_prompt:
            self.formatter.context = self.system_prompt

        return self

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        # TODO: the ReAct formatter does not explicitly specify PromptTemplate
        # objects, but wrap it in this to obey the interface
        react_header = self.formatter.system_header
        return {"react_header": PromptTemplate(react_header)}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "react_header" in prompts:
            react_header = prompts["react_header"]
            if isinstance(react_header, str):
                react_header = PromptTemplate(react_header)
            self.formatter.system_header = react_header.format()

    async def _get_response(self, current_llm_input: List[ChatMessage]) -> ChatResponse:
        return await self.llm.achat(current_llm_input)

    async def _get_streaming_response(
        self, ctx: Context, current_llm_input: List[ChatMessage]
    ) -> ChatResponse:
        response = await self.llm.astream_chat(
            current_llm_input,
        )

        # last_chat_response will be used later, after the loop.
        # We initialize it so it's valid even when 'response' is empty
        last_chat_response = ChatResponse(message=ChatMessage())
        async for last_chat_response in response:
            raw = (
                last_chat_response.raw.model_dump()
                if isinstance(last_chat_response.raw, BaseModel)
                else last_chat_response.raw
            )
            # some code paths (namely react agent via llm.predict_and_call for non function calling llms) pass through a context without starting the workflow.
            # They do so in order to conform to the interface, and share state between tools, however the events are discarded and not exposed to the caller,
            # so just don't write events if the context is not running.
            if ctx.is_running:
                ctx.write_event_to_stream(
                    AgentStream(
                        delta=last_chat_response.delta or "",
                        response=last_chat_response.message.content or "",
                        raw=raw,
                        current_agent_name=self.name,
                        thinking_delta=last_chat_response.additional_kwargs.get(
                            "thinking_delta", None
                        ),
                    )
                )

        return last_chat_response

    async def take_step(
        self,
        ctx: Context,
        llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
        memory: BaseMemory,
    ) -> AgentOutput:
        """Take a single step with the React agent."""
        # remove system prompt, since the react prompt will be combined with it
        if llm_input[0].role == "system":
            system_prompt = llm_input[0].content or ""
            llm_input = llm_input[1:]
        else:
            system_prompt = ""

        output_parser = self.output_parser
        react_chat_formatter = self.formatter

        # Format initial chat input
        current_reasoning: list[BaseReasoningStep] = await ctx.store.get(
            self.reasoning_key, default=[]
        )
        input_chat = react_chat_formatter.format(
            tools,
            chat_history=llm_input,
            current_reasoning=current_reasoning,
        )
        # some code paths (namely react agent via llm.predict_and_call for non function calling llms) pass through a context without starting the workflow.
        # They do so in order to conform to the interface, and share state between tools, however the events are discarded and not exposed to the caller,
        # so just don't write events if the context is not running.
        if ctx.is_running:
            ctx.write_event_to_stream(
                AgentInput(input=input_chat, current_agent_name=self.name)
            )

        # Initial LLM call
        if self.streaming:
            last_chat_response = await self._get_streaming_response(ctx, input_chat)
        else:
            last_chat_response = await self._get_response(input_chat)

        # Parse reasoning step and check if done
        message_content = last_chat_response.message.content
        if not message_content:
            raise ValueError("Got empty message")

        try:
            reasoning_step = output_parser.parse(message_content, is_streaming=False)
        except ValueError as e:
            error_msg = (
                f"Error while parsing the output: {e!s}\n\n"
                "The output should be in one of the following formats:\n"
                "1. To call a tool:\n"
                "```\n"
                "Thought: <thought>\n"
                "Action: <action>\n"
                "Action Input: <action_input>\n"
                "```\n"
                "2. To answer the question:\n"
                "```\n"
                "Thought: <thought>\n"
                "Answer: <answer>\n"
                "```\n"
            )

            raw = (
                last_chat_response.raw.model_dump()
                if isinstance(last_chat_response.raw, BaseModel)
                else last_chat_response.raw
            )
            # Return with retry messages to let the LLM fix the error
            return AgentOutput(
                response=last_chat_response.message,
                raw=raw,
                current_agent_name=self.name,
                retry_messages=[
                    last_chat_response.message,
                    ChatMessage(role="user", content=error_msg),
                ],
            )

        # add to reasoning if not a handoff
        current_reasoning.append(reasoning_step)
        await ctx.store.set(self.reasoning_key, current_reasoning)

        # If response step, we're done
        raw = (
            last_chat_response.raw.model_dump()
            if isinstance(last_chat_response.raw, BaseModel)
            else last_chat_response.raw
        )
        if reasoning_step.is_done:
            return AgentOutput(
                response=last_chat_response.message,
                raw=raw,
                current_agent_name=self.name,
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
            response=last_chat_response.message,
            tool_calls=tool_calls,
            raw=raw,
            current_agent_name=self.name,
        )

    async def handle_tool_call_results(
        self, ctx: Context, results: List[ToolCallResult], memory: BaseMemory
    ) -> None:
        """Handle tool call results for React agent."""
        current_reasoning: list[BaseReasoningStep] = await ctx.store.get(
            self.reasoning_key, default=[]
        )
        for tool_call_result in results:
            obs_step = ObservationReasoningStep(
                observation=str(tool_call_result.tool_output.content),
                return_direct=tool_call_result.return_direct,
            )
            current_reasoning.append(obs_step)

            if (
                tool_call_result.return_direct
                and tool_call_result.tool_name != "handoff"
            ):
                current_reasoning.append(
                    ResponseReasoningStep(
                        thought=obs_step.observation,
                        response=obs_step.observation,
                        is_streaming=False,
                    )
                )
                break

        await ctx.store.set(self.reasoning_key, current_reasoning)

    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        """Finalize the React agent."""
        current_reasoning: list[BaseReasoningStep] = await ctx.store.get(
            self.reasoning_key, default=[]
        )

        if len(current_reasoning) > 0 and isinstance(
            current_reasoning[-1], ResponseReasoningStep
        ):
            reasoning_str = "\n".join([x.get_content() for x in current_reasoning])

            if reasoning_str:
                reasoning_msg = ChatMessage(role="assistant", content=reasoning_str)
                await memory.aput(reasoning_msg)
                await ctx.store.set(self.reasoning_key, [])

            # Find the text block in the response to modify it directly
            text_block = None
            for block in output.response.blocks:
                if isinstance(block, TextBlock):
                    text_block = block
                    break

            # remove "Answer:" from the response (now checking text_block.text)
            if text_block and "Answer:" in text_block.text:
                start_idx = text_block.text.find("Answer:")
                if start_idx != -1:
                    # Modify the .text attribute of the block, NOT response.content
                    text_block.text = text_block.text[
                        start_idx + len("Answer:") :
                    ].strip()

            # clear scratchpad
            await ctx.store.set(self.reasoning_key, [])

        return output
