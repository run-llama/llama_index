import json
from typing import Optional, Sequence

from llama_index.agent.utils import FunctionMessage, monkey_patch_langchain

# TODO: right now langchain does not support function messages
#       monkey patch it to support it
monkey_patch_langchain()

from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory

from llama_index.callbacks.base import CallbackManager
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.tools import BaseTool

DEFAULT_MAX_FUNCTION_CALLS = 5


class OpenAIAgent(BaseChatEngine, BaseQueryEngine):
    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: ChatOpenAI,
        chat_history: ChatMessageHistory,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._chat_history = chat_history
        self._verbose = verbose
        self._max_function_calls = max_function_calls
        self.callback_manager = callback_manager or CallbackManager([])

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

        n_function_calls = 0
        function_call = ai_message.additional_kwargs.get("function_call", None)
        while function_call is not None:
            if n_function_calls >= self._max_function_calls:
                print(f"Exceeded max function calls: {self._max_function_calls}.")
                continue

            function_output = self._call_function(function_call)
            n_function_calls += 1

            function_message = FunctionMessage(
                content=function_output, name=function_call["name"]
            )
            chat_history.add_message(function_message)

            # send function call & output back to get another response
            ai_message = self._llm.predict_messages(chat_history.messages)
            function_call = ai_message.additional_kwargs.get("function_call", None)

        chat_history.add_message(ai_message)

        return Response(ai_message.content)

    async def achat(
        self, message: str, chat_history: Optional[ChatMessageHistory] = None
    ) -> RESPONSE_TYPE:
        chat_history = chat_history or self._chat_history
        chat_history.add_user_message(message)
        functions = [tool.metadata.to_openai_function() for tool in self._tools]

        # TODO: Support forced function call
        ai_message = await self._llm.apredict_messages(
            chat_history.messages, functions=functions
        )
        chat_history.add_message(ai_message)

        n_function_calls = 0
        function_call = ai_message.additional_kwargs.get("function_call", None)
        while function_call is not None:
            if n_function_calls >= self._max_function_calls:
                print(f"Exceeded max function calls: {self._max_function_calls}.")
                continue

            function_output = self._call_function(function_call)
            n_function_calls += 1

            function_message = FunctionMessage(
                content=function_output, name=function_call["name"]
            )
            chat_history.add_message(function_message)

            # send function back to get a natural language response
            ai_message = await self._llm.apredict_messages(chat_history.messages)
            function_call = ai_message.additional_kwargs.get("function_call", None)

        chat_history.add_message(ai_message)

        return Response(ai_message.content)

    def _get_function_by_name(self, name: str) -> BaseTool:
        name_to_tool = {tool.metadata.name: tool for tool in self._tools}
        if name not in name_to_tool:
            raise ValueError(f"Tool with name {name} not found")
        return name_to_tool[name]

    def _call_function(self, function_call: dict) -> str:
        """Call a function and return the output as a string."""
        name = function_call["name"]
        arguments_str = function_call["arguments"]
        if self._verbose:
            print("=== Calling Function ===")
            print(f"Calling function: {name} with args: {arguments_str}")
        tool = self._get_function_by_name(name)
        argument_dict = json.loads(arguments_str)
        output = tool(**argument_dict)
        if self._verbose:
            print(f"Got output: {output}")
            print("========================")
        return str(output)

    # ===== Query Engine Interface =====
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self.chat(
            query_bundle.query_str,
            chat_history=ChatMessageHistory(),
        )

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return await self.achat(
            query_bundle.query_str,
            chat_history=ChatMessageHistory(),
        )
