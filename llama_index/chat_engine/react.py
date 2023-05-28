from typing import Optional

from llama_index.chat_engine.types import BaseChatEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.langchain_helpers.agents.agents import (AgentExecutor,
                                                         AgentType,
                                                         initialize_agent)
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.tools.query_engine import QueryEngineTool


class ReActChatEngine(BaseChatEngine):
    def __init__(
        self,
        query_engine_tool: QueryEngineTool,
        service_context: Optional[ServiceContext] = None,
    ) -> None:
        self._query_engine_tool = query_engine_tool
        self._service_context = service_context or ServiceContext.from_defaults()
        self._agent = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        tools = [self._query_engine_tool.as_langchain_tool()]
        assert isinstance(self._service_context.llm_predictor, LLMPredictor)
        self._agent = initialize_agent(
            tools=tools,
            llm=self._service_context.llm_predictor.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        )

    def chat(self, message: str) -> RESPONSE_TYPE:
        self._agent.run(input=message)

    def achat(self, message: str) -> RESPONSE_TYPE:
        self._agent.arun(input=message)
