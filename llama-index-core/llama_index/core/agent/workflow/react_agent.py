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
from llama_index.core.agent.workflow.single_agent_workflow import SingleAgentRunnerMixin
from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCallResult,
)
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.bridge.pydantic import BaseModel, Field
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


class ReActAgent(SingleAgentRunnerMixin, BaseWorkflowAgent):
    """React agent implementation."""

    reasoning_key: str = "current_reasoning"
    output_parser: ReActOutputParser = Field(
        default_factory=ReActOutputParser, description="The react output parser"
    )
    formatter: ReActChatFormatter = Field(
        default_factory=default_formatter,
        description="The react chat formatter to format the reasoning steps and chat history into an llm input.",
    )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        # TODO: the ReAct formatter does not explicitly specify PromptTemplate
        # objects, but wrap it in this to obey the interface
        react_header = self.formatter.system_header
        return {"react_header": PromptTemplate(react_header)}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "system_prompt" in prompts:
            react_header = cast(PromptTemplate, prompts["react_header"])
            self.formatter.system_header = react_header.template

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
        react_chat_formatter.context = system_prompt

        # Format initial chat input
        current_reasoning: list[BaseReasoningStep] = await ctx.get(
            self.reasoning_key, default=[]
        )
        input_chat = react_chat_formatter.format(
            tools,
            chat_history=llm_input,
            current_reasoning=current_reasoning,
        )
        ctx.write_event_to_stream(
            AgentInput(input=input_chat, current_agent_name=self.name)
        )

        # Initial LLM call
        response = await self.llm.astream_chat(input_chat)
        # last_chat_response will be used later, after the loop.
        # We initialize it so it's valid even when 'response' is empty
        last_chat_response = ChatResponse(message=ChatMessage())
        async for last_chat_response in response:
            raw = (
                last_chat_response.raw.model_dump()
                if isinstance(last_chat_response.raw, BaseModel)
                else last_chat_response.raw
            )
            ctx.write_event_to_stream(
                AgentStream(
                    delta=last_chat_response.delta or "",
                    response=last_chat_response.message.content or "",
                    tool_calls=[],
                    raw=raw,
                    current_agent_name=self.name,
                )
            )

        # Parse reasoning step and check if done
        message_content = last_chat_response.message.content
        if not message_content:
            raise ValueError("Got empty message")

        try:
            reasoning_step = output_parser.parse(message_content, is_streaming=False)
        except ValueError as e:
            error_msg = f"Error: Could not parse output. Please follow the thought-action-input format. Try again. Details: {e!s}"
            await memory.aput(last_chat_response.message)
            await memory.aput(ChatMessage(role="user", content=error_msg))

            raw = (
                last_chat_response.raw.model_dump()
                if isinstance(last_chat_response.raw, BaseModel)
                else last_chat_response.raw
            )
            return AgentOutput(
                response=last_chat_response.message,
                tool_calls=[],
                raw=raw,
                current_agent_name=self.name,
            )

        # add to reasoning if not a handoff
        current_reasoning.append(reasoning_step)
        await ctx.set(self.reasoning_key, current_reasoning)

        # If response step, we're done
        raw = (
            last_chat_response.raw.model_dump()
            if isinstance(last_chat_response.raw, BaseModel)
            else last_chat_response.raw
        )
        if reasoning_step.is_done:
            return AgentOutput(
                response=last_chat_response.message,
                tool_calls=[],
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
        current_reasoning: list[BaseReasoningStep] = await ctx.get(
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

        await ctx.set(self.reasoning_key, current_reasoning)

    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        """Finalize the React agent."""
        current_reasoning: list[BaseReasoningStep] = await ctx.get(
            self.reasoning_key, default=[]
        )

        if len(current_reasoning) > 0 and isinstance(
            current_reasoning[-1], ResponseReasoningStep
        ):
            reasoning_str = "\n".join([x.get_content() for x in current_reasoning])

            if reasoning_str:
                reasoning_msg = ChatMessage(role="assistant", content=reasoning_str)
                await memory.aput(reasoning_msg)
                await ctx.set(self.reasoning_key, [])

            # remove "Answer:" from the response
            if output.response.content and "Answer:" in output.response.content:
                start_idx = output.response.content.find("Answer:")
                if start_idx != -1:
                    output.response.content = output.response.content[
                        start_idx + len("Answer:") :
                    ].strip()

            # clear scratchpad
            await ctx.set(self.reasoning_key, [])

        return output
