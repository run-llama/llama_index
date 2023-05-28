from typing import Any, List, Optional, Tuple

from llama_index.chat_engine.types import BaseChatEngine
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.langchain_helpers.agents.agents import (
    AgentExecutor,
    AgentType,
    initialize_agent,
)
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.tools.query_engine import QueryEngineTool
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory


class ReActChatEngine(BaseChatEngine):
    """ReAct Chat Engine.


    Use a ReAct agent loop with query engine tools. Implemented via LangChain agent.
    """

    def __init__(
        self,
        query_engine_tools: List[QueryEngineTool],
        service_context: Optional[ServiceContext] = None,
        memory: Optional[BaseChatMemory] = None,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        verbose: bool = False,
    ) -> None:
        self._query_engine_tools = query_engine_tools
        self._service_context = service_context or ServiceContext.from_defaults()
        if memory is None:
            memory = ConversationBufferMemory(memory_key="chat_history")
            if chat_history is not None:
                for human_message, ai_message in chat_history:
                    memory.chat_memory.add_user_message(human_message)
                    memory.chat_memory.add_ai_message(ai_message)

        self._memory = memory
        self._verbose = verbose

        self._agent = self._create_agent()

    @classmethod
    def from_defaults(
        cls,
        query_engine: BaseQueryEngine,
        name: Optional[str] = None,
        description: Optional[str] = None,
        service_context: Optional[ServiceContext] = None,
        memory: Optional[BaseChatMemory] = None,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "ReActChatEngine":
        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine, name=name, description=description
        )
        return cls(
            query_engine_tools=[query_engine_tool],
            service_context=service_context,
            memory=memory,
            chat_history=chat_history,
            verbose=verbose,
        )

    def _create_agent(self) -> AgentExecutor:
        tools = [qe_tool.as_langchain_tool() for qe_tool in self._query_engine_tools]
        if not isinstance(self._service_context.llm_predictor, LLMPredictor):
            raise ValueError("Currently only supports LangChain based LLMPredictor.")
        return initialize_agent(
            tools=tools,
            llm=self._service_context.llm_predictor.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self._memory,
            verbose=self._verbose,
        )

    def chat(self, message: str) -> RESPONSE_TYPE:
        response = self._agent.run(input=message)
        return Response(response=response)

    async def achat(self, message: str) -> RESPONSE_TYPE:
        response = await self._agent.arun(input=message)
        return Response(response=response)

    def reset(self) -> None:
        self._agent = self._create_agent()
