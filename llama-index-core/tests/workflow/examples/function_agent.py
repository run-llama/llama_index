import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional, Union

from llama_index.core.llms import ChatMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import BaseTool, FunctionTool, ToolOutput, ToolSelection
from llama_index.core.workflow import (
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


class FuncationCallingAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: Optional[FunctionCallingLLM] = None,
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []

        self.llm = llm or OpenAI()
        assert self.llm.metadata.is_function_calling_model

        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)
        self.sources = []

    @step()
    async def prepare_chat_history(self, ev: StartEvent) -> InputEvent:
        # clear sources
        self.sources = []

        # get user input
        user_input = ev.get("input")
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.put(user_msg)

        # get chat history
        chat_history = self.memory.get()
        return InputEvent(input=chat_history)

    @step()
    async def handle_llm_input(self, ev: InputEvent) -> Union[ToolCallEvent, StopEvent]:
        chat_history = ev.input

        response = await self.llm.achat_with_tools(
            self.tools, chat_history=chat_history
        )
        self.memory.put(response.message)

        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            return StopEvent(result={"response": response, "sources": [*self.sources]})
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step()
    async def handle_tool_calls(
        self, ev: ToolCallEvent
    ) -> Union[InputEvent, StopEvent]:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                self.sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        for msg in tool_msgs:
            self.memory.put(msg)

        chat_history = self.memory.get()
        return InputEvent(input=chat_history)


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

    agent = FuncationCallingAgent(
        llm=OpenAI(model="gpt-4o-mini"), tools=tools, timeout=120, verbose=False
    )

    print("Hello!")
    ret = await agent.run(input="Hello!")
    print(ret["response"])

    print("What is (2123 + 2321) * 312?")
    ret = await agent.run(input="What is (2123 + 2321) * 312?")
    print(ret["response"])

    draw_all_possible_flows(agent)
    draw_most_recent_execution(agent)


if __name__ == "__main__":
    asyncio.run(main())
