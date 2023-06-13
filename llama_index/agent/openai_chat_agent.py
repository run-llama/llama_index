import json
from typing import Any, Optional, Sequence

from langchain.chat_models import ChatOpenAI
from langchain.schema import FunctionMessage
from pydantic import BaseModel

from llama_index.agent.types import BaseChatAgent
from llama_index.chat_engine.types import ChatHistoryType
from llama_index.chat_engine.utils import to_langchain_chat_history
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.tools.types import BaseTool


class FunctionCall(BaseModel):
    name: str
    arguments: Optional[dict] = None


class OpenAIChatAgent(BaseChatAgent):
    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: ChatOpenAI,
        chat_history: ChatHistoryType,
        verbose: bool = False,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._chat_history = chat_history
        self._verbose = verbose

    @classmethod
    def from_tools(
        cls,
        tools: Sequence[BaseTool],
        llm: Optional[ChatOpenAI] = None,
        chat_history: Optional[ChatHistoryType] = None,
        verbose: bool = False,
    ) -> "OpenAIChatAgent":
        chat_history = chat_history or []
        llm = llm or ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-next")
        if not isinstance(llm, ChatOpenAI):
            raise ValueError("llm must be a ChatOpenAI instance")

        return cls(tools=tools, llm=llm, chat_history=chat_history, verbose=verbose)

    def reset(self) -> None:
        self._chat_history = []

    async def achat(self, message: str) -> RESPONSE_TYPE:
        # TODO: implement
        return self.chat(message)

    def chat(
        self, message: str, chat_history: Optional[ChatHistoryType] = None
    ) -> RESPONSE_TYPE:
        if chat_history is None:
            chat_history = self._chat_history

        # prepare chat history
        lc_chat_history = to_langchain_chat_history(chat_history)

        # prepare user message
        lc_chat_history.add_user_message(message)
        functions = [tool.metadata.to_openai_function() for tool in self._tools]

        # TODO: Support forced function call
        ai_message = self._llm.predict_messages(
            lc_chat_history.messages, functions=functions
        )
        lc_chat_history.add_message(ai_message)

        function_call = ai_message.additional_kwargs.get("function_call", None)
        if function_call is not None:
            # Call function
            # json_dict = json.loads(function_call)
            print(function_call)
            function_output = self._call_function(function_call)

            # TODO: properly support function message
            # TODO: right now requires dirty changes in langchain library
            function_message = FunctionMessage(
                role="function", content=function_output, name=function_call["name"]
            )
            lc_chat_history.add_message(function_message)

            # send function back to get a natural language response
            ai_message = self._llm.predict_messages(lc_chat_history.messages)

        # Record exchange
        self._chat_history.append((message, ai_message.content))

        return Response(ai_message.content)

    def query(self, query: str) -> Any:
        return self.chat(query, chat_history=[])

    def _get_tool(self, name: str) -> BaseTool:
        name_to_tool = {tool.metadata.name: tool for tool in self._tools}
        if name not in name_to_tool:
            raise ValueError(f"Tool with name {name} not found")
        return name_to_tool[name]

    def _call_function(self, function_call: dict) -> str:
        name = function_call["name"]
        arguments_str = function_call["arguments"]
        if self._verbose:
            print(f"Calling function: {name} with args: {arguments_str}")
        tool = self._get_tool(name)
        argument_dict = json.loads(arguments_str)
        output = tool(**argument_dict)
        return str(output)
