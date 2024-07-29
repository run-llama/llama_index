import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional, Union

from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import BaseTool, FunctionTool, ToolOutput, ToolSelection
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    draw_all_possible_flows,
    draw_most_recent_execution,
)
from llama_index.llms.openai import OpenAI


@dataclass
class PrepEvent(Event):
    pass


@dataclass
class InputEvent(Event):
    input: List[ChatMessage]


@dataclass
class ToolCallEvent(Event):
    tool_calls: List[ToolSelection]


@dataclass
class FunctionOutputEvent(Event):
    output: ToolOutput


@dataclass
class FinalizeStepEvent(Event):
    pass


class ReActAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: Optional[FunctionCallingLLM] = None,
        tools: Optional[List[BaseTool]] = None,
        extra_context: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []

        self.llm = llm or OpenAI()
        assert self.llm.metadata.is_function_calling_model

        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)
        self.formatter = ReActChatFormatter(context=extra_context or "")
        self.output_parser = ReActOutputParser()
        self.sources = []

    @step(pass_context=True)
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        # clear sources
        self.sources = []

        # get user input
        user_input = ev.get("input")
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.put(user_msg)

        # clear current reasoning
        ctx["current_reasoning"] = []

        return PrepEvent()

    @step(pass_context=True)
    async def prepare_chat_history(self, ctx: Context, ev: PrepEvent) -> InputEvent:
        # get chat history
        chat_history = self.memory.get()
        current_reasoning = ctx.get("current_reasoning", [])
        llm_input = self.formatter.format(
            self.tools, chat_history, current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input)

    @step(pass_context=True)
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> Union[ToolCallEvent, StopEvent]:
        chat_history = ev.input

        response = await self.llm.achat(chat_history)

        import pdb

        pdb.set_trace()
        try:
            reasoning_step = self.output_parser.parse(response.message.content)
            ctx.get("current_reasoning", []).append(reasoning_step)
            if reasoning_step.is_done:
                self.memory.put(
                    ChatMessage(role="assistant", content=reasoning_step.response)
                )
                return StopEvent(
                    result={
                        "response": response,
                        "sources": [*self.sources],
                        "reasoning": ctx.get("current_reasoning", []),
                    }
                )
            elif isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake", tool_name=tool_name, tool_kwargs=tool_args
                        )
                    ]
                )
        except Exception as e:
            ctx.get("current_reasoning", []).append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}"
                )
            )

        # if no tool calls or final response, iterate again
        return PrepEvent()

    @step(pass_context=True)
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> Union[InputEvent, StopEvent]:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                ctx.get("current_reasoning", []).append(
                    ObservationReasoningStep(
                        observation=f"Tool {tool_call.tool_name} does not exist"
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                self.sources.append(tool_output)
                ctx.get("current_reasoning", []).append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
            except Exception as e:
                ctx.get("current_reasoning", []).append(
                    ObservationReasoningStep(
                        observation=f"Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )

        # prep the next iteraiton
        return PrepEvent()


async def main() -> None:
    def add(x: int, y: int) -> int:
        """Useful function to add two numbers."""
        return x + y

    def multiply(x: int, y: int) -> int:
        """Useful function to multiply two numbers."""
        return x * y

    tools = [
        FunctionTool.from_defaults(add),
        FunctionTool.from_defaults(multiply),
    ]

    agent = ReActAgent(
        llm=OpenAI(model="gpt-4-turbo", temperature=0.25),
        tools=tools,
        timeout=120,
        verbose=False,
    )

    print("Hello!")
    ret = await agent.run(input="Hello!")
    print(ret["response"])

    print("What is (2123 + 2321) * 312?")
    ret = await agent.run(input="What is (2123 + 2321) * 312?")
    print(ret["response"])
    print(ret["reasoning"])

    draw_all_possible_flows(agent)
    draw_most_recent_execution(agent)


if __name__ == "__main__":
    asyncio.run(main())
