import uuid
from typing import List, cast

from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCallResult,
)
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.memory import BaseMemory
from llama_index.core.tools import AsyncBaseTool
from llama_index.core.workflow import Context


class ReactAgent(BaseWorkflowAgent):
    """React agent implementation."""

    async def take_step(
        self,
        ctx: Context,
        llm_input: List[ChatMessage],
        tools: List[AsyncBaseTool],
        memory: BaseMemory,
    ) -> AgentOutput:
        """Take a single step with the React agent."""
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

        ctx.write_event_to_stream(
            AgentInput(input=input_chat, current_agent_name=self.name)
        )

        # Initial LLM call
        response = await self.llm.astream_chat(input_chat)
        async for r in response:
            ctx.write_event_to_stream(
                AgentStream(
                    delta=r.delta or "",
                    tool_calls=[],
                    raw_response=r.raw,
                    current_agent_name=self.name,
                )
            )

        # Parse reasoning step and check if done
        message_content = r.message.content
        if not message_content:
            raise ValueError("Got empty message")

        try:
            reasoning_step = output_parser.parse(message_content, is_streaming=False)
        except ValueError as e:
            error_msg = f"Error: Could not parse output. Please follow the thought-action-input format. Try again. Details: {e!s}"
            await memory.aput(r.message)
            await memory.aput(ChatMessage(role="user", content=error_msg))

            return AgentOutput(
                response=r.message.content,
                tool_calls=[],
                raw_response=r.raw,
                current_agent_name=self.name,
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
            response=r.message.content,
            tool_calls=tool_calls,
            raw_response=r.raw,
            current_agent_name=self.name,
        )

    async def handle_tool_call_results(
        self, ctx: Context, results: List[ToolCallResult], memory: BaseMemory
    ) -> None:
        """Handle tool call results for React agent."""
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

    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        """Finalize the React agent."""
        current_reasoning: list[BaseReasoningStep] = await ctx.get(
            "current_reasoning", default=[]
        )

        reasoning_str = "\n".join([x.get_content() for x in current_reasoning])
        reasoning_msg = ChatMessage(role="assistant", content=reasoning_str)

        await memory.aput(reasoning_msg)
        await ctx.set("current_reasoning", [])

        # remove "Answer:" from the response
        if output.response and "Answer:" in output.response:
            start_idx = output.response.index("Answer:")
            output.response = output.response[start_idx + len("Answer:") :].strip()

        return output
