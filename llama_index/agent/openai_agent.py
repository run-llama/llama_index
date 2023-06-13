import json
from typing import Any, Optional, Sequence

from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.schema import FunctionMessage

from llama_index.chat_engine.types import BaseChatEngine
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.tools.types import BaseTool


class OpenAIAgent(BaseChatEngine, BaseQueryEngine):
    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: ChatOpenAI,
        chat_history: ChatMessageHistory,
        verbose: bool = False,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._chat_history = chat_history
        self._verbose = verbose

    @property
    def chat_history(self) -> ChatMessageHistory:
        return self._chat_history

    @classmethod
    def from_tools(
        cls,
        tools: Optional[Sequence[BaseTool]] = None,
        llm: Optional[ChatOpenAI] = None,
        chat_history: Optional[ChatMessageHistory] = None,
        verbose: bool = False,
    ) -> "OpenAIAgent":
        tools = tools or []
        lc_chat_history = chat_history or ChatMessageHistory()
        llm = llm or ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-next")
        if not isinstance(llm, ChatOpenAI):
            raise ValueError("llm must be a ChatOpenAI instance")

        return cls(tools=tools, llm=llm, chat_history=lc_chat_history, verbose=verbose)

    def reset(self) -> None:
        self._chat_history.clear()

    def chat(
        self, message: str, chat_history: Optional[ChatMessageHistory] = None
    ) -> RESPONSE_TYPE:
        chat_history = chat_history or self._chat_history
        chat_history.add_user_message(message)
        functions = [tool.metadata.to_openai_function() for tool in self._tools]

        # TODO: Support forced function call
        ai_message = self._llm.predict_messages(
            chat_history.messages, functions=functions
        )
        chat_history.add_message(ai_message)

        function_call = ai_message.additional_kwargs.get("function_call", None)
        if function_call is not None:
            function_output = self._call_function(function_call)

            # TODO: properly support function message,
            #       right now requires dirty changes in langchain library
            function_message = FunctionMessage(
                content=function_output, name=function_call["name"]
            )
            chat_history.add_message(function_message)

            # send function back to get a natural language response
            ai_message = self._llm.predict_messages(chat_history.messages)

        chat_history.add_message(ai_message)

        return Response(ai_message.content)

    async def achat(self, message: str) -> RESPONSE_TYPE:
        chat_history = chat_history or self._chat_history
        chat_history.add_user_message(message)
        functions = [tool.metadata.to_openai_function() for tool in self._tools]

        # TODO: Support forced function call
        ai_message = await self._llm.apredict_messages(
            chat_history.messages, functions=functions
        )
        chat_history.add_message(ai_message)

        function_call = ai_message.additional_kwargs.get("function_call", None)
        if function_call is not None:
            function_output = self._call_function(function_call)

            # TODO: properly support function message,
            #       right now requires dirty changes in langchain library
            function_message = FunctionMessage(
                content=function_output, name=function_call["name"]
            )
            chat_history.add_message(function_message)

            # send function back to get a natural language response
            ai_message = await self._llm.apredict_messages(chat_history.messages)

        chat_history.add_message(ai_message)

        return Response(ai_message.content)

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

    # ===== Query Engine Interface =====
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self.chat(
            query_bundle.query_str,
            chat_history=[],
        )

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return await self.achat(
            query_bundle.query_str,
            chat_history=[],
        )
