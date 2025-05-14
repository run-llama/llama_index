"""SQL Vector query engine."""

import logging
from typing import Any, Optional, Union

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.struct_store.sql_query import (
    BaseSQLTableQueryEngine,
    NLSQLTableQueryEngine,
)
from llama_index.core.indices.vector_store.retrievers.auto_retriever import (
    VectorIndexAutoRetriever,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index.core.query_engine.sql_join_query_engine import (
    SQLAugmentQueryTransform,
    SQLJoinQueryEngine,
)
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
from llama_index.core.selectors.pydantic_selectors import PydanticSingleSelector
from llama_index.core.tools.query_engine import QueryEngineTool

logger = logging.getLogger(__name__)


DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT_TMPL = """
The original question is given below.
This question has been translated into a SQL query. \
Both the SQL query and the response are given below.
Given the SQL response, the question has also been translated into a vector store query.
The vector store query and response is given below.
Given SQL query, SQL response, transformed vector store query, and vector store \
response, please synthesize a response to the original question.

Original question: {query_str}
SQL query: {sql_query_str}
SQL response: {sql_response_str}
Transformed vector store query: {query_engine_query_str}
Vector store response: {query_engine_response_str}
Response:
"""
DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT_TMPL
)


# NOTE: maintain for backwards compatibility
class SQLAutoVectorQueryEngine(SQLJoinQueryEngine):
    """
    SQL + Vector Index Auto Retriever Query Engine.

    This query engine can query both a SQL database
    as well as a vector database. It will first decide
    whether it needs to query the SQL database or vector store.
    If it decides to query the SQL database, it will also decide
    whether to augment information with retrieved results from the vector store.
    We use the VectorIndexAutoRetriever to retrieve results.

    Args:
        sql_query_tool (QueryEngineTool): Query engine tool for SQL database.
        vector_query_tool (QueryEngineTool): Query engine tool for vector database.
        selector (Optional[Union[LLMSingleSelector, PydanticSingleSelector]]):
            Selector to use.
        sql_vector_synthesis_prompt (Optional[BasePromptTemplate]):
            Prompt to use for SQL vector synthesis.
        sql_augment_query_transform (Optional[SQLAugmentQueryTransform]): Query
            transform to use for SQL augmentation.
        use_sql_vector_synthesis (bool): Whether to use SQL vector synthesis.
        callback_manager (Optional[CallbackManager]): Callback manager to use.
        verbose (bool): Whether to print intermediate results.

    """

    def __init__(
        self,
        sql_query_tool: QueryEngineTool,
        vector_query_tool: QueryEngineTool,
        selector: Optional[Union[LLMSingleSelector, PydanticSingleSelector]] = None,
        llm: Optional[LLM] = None,
        sql_vector_synthesis_prompt: Optional[BasePromptTemplate] = None,
        sql_augment_query_transform: Optional[SQLAugmentQueryTransform] = None,
        use_sql_vector_synthesis: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize params."""
        # validate that the query engines are of the right type
        if not isinstance(
            sql_query_tool.query_engine,
            (BaseSQLTableQueryEngine, NLSQLTableQueryEngine),
        ):
            raise ValueError(
                "sql_query_tool.query_engine must be an instance of "
                "BaseSQLTableQueryEngine or NLSQLTableQueryEngine"
            )
        if not isinstance(vector_query_tool.query_engine, RetrieverQueryEngine):
            raise ValueError(
                "vector_query_tool.query_engine must be an instance of "
                "RetrieverQueryEngine"
            )
        if not isinstance(
            vector_query_tool.query_engine.retriever, VectorIndexAutoRetriever
        ):
            raise ValueError(
                "vector_query_tool.query_engine.retriever must be an instance "
                "of VectorIndexAutoRetriever"
            )

        sql_vector_synthesis_prompt = (
            sql_vector_synthesis_prompt or DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT
        )
        super().__init__(
            sql_query_tool,
            vector_query_tool,
            selector=selector,
            llm=llm,
            sql_join_synthesis_prompt=sql_vector_synthesis_prompt,
            sql_augment_query_transform=sql_augment_query_transform,
            use_sql_join_synthesis=use_sql_vector_synthesis,
            callback_manager=callback_manager,
            verbose=verbose,
        )

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {
            "selector": self._selector,
            "sql_augment_query_transform": self._sql_augment_query_transform,
        }

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"sql_join_synthesis_prompt": self._sql_join_synthesis_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "sql_join_synthesis_prompt" in prompts:
            self._sql_join_synthesis_prompt = prompts["sql_join_synthesis_prompt"]

    @classmethod
    def from_sql_and_vector_query_engines(
        cls,
        sql_query_engine: Union[BaseSQLTableQueryEngine, NLSQLTableQueryEngine],
        sql_tool_name: str,
        sql_tool_description: str,
        vector_auto_retriever: RetrieverQueryEngine,
        vector_tool_name: str,
        vector_tool_description: str,
        selector: Optional[Union[LLMSingleSelector, PydanticSingleSelector]] = None,
        **kwargs: Any,
    ) -> "SQLAutoVectorQueryEngine":
        """
        From SQL and vector query engines.

        Args:
            sql_query_engine (BaseSQLTableQueryEngine): SQL query engine.
            vector_query_engine (VectorIndexAutoRetriever): Vector retriever.
            selector (Optional[Union[LLMSingleSelector, PydanticSingleSelector]]):
                Selector to use.

        """
        sql_query_tool = QueryEngineTool.from_defaults(
            sql_query_engine, name=sql_tool_name, description=sql_tool_description
        )
        vector_query_tool = QueryEngineTool.from_defaults(
            vector_auto_retriever,
            name=vector_tool_name,
            description=vector_tool_description,
        )
        return cls(sql_query_tool, vector_query_tool, selector, **kwargs)
