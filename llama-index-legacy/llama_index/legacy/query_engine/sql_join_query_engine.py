"""SQL Join query engine."""

import logging
from typing import Callable, Dict, Optional, Union

from llama_index.callbacks.base import CallbackManager
from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.core.response.schema import RESPONSE_TYPE, Response
from llama_index.indices.query.query_transform.base import BaseQueryTransform
from llama_index.indices.struct_store.sql_query import (
    BaseSQLTableQueryEngine,
    NLSQLTableQueryEngine,
)
from llama_index.llm_predictor.base import LLMPredictorType
from llama_index.llms.utils import resolve_llm
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.schema import QueryBundle
from llama_index.selectors.llm_selectors import LLMSingleSelector
from llama_index.selectors.pydantic_selectors import PydanticSingleSelector
from llama_index.selectors.utils import get_selector_from_context
from llama_index.service_context import ServiceContext
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.utils import print_text

logger = logging.getLogger(__name__)


DEFAULT_SQL_JOIN_SYNTHESIS_PROMPT_TMPL = """
The original question is given below.
This question has been translated into a SQL query. Both the SQL query and \
the response are given below.
Given the SQL response, the question has also been transformed into a more \
detailed query,
and executed against another query engine.
The transformed query and query engine response are also given below.
Given SQL query, SQL response, transformed query, and query engine response, \
please synthesize a response to the original question.

Original question: {query_str}
SQL query: {sql_query_str}
SQL response: {sql_response_str}
Transformed query: {query_engine_query_str}
Query engine response: {query_engine_response_str}
Response:
"""
DEFAULT_SQL_JOIN_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_SQL_JOIN_SYNTHESIS_PROMPT_TMPL
)


DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT_TMPL = """
"The original question is given below.
This question has been translated into a SQL query. Both the SQL query and the \
response are given below.
The SQL response either answers the question, or should provide additional context \
that can be used to make the question more specific.
Your job is to come up with a more specific question that needs to be answered to \
fully answer the original question, or 'None' if the original question has already \
been fully answered from the SQL response. Do not create a new question that is \
irrelevant to the original question; in that case return None instead.

Examples:

Original question: Please give more details about the demographics of the city with \
the highest population.
SQL query: SELECT city, population FROM cities ORDER BY population DESC LIMIT 1
SQL response: The city with the highest population is New York City.
New question: Can you tell me more about the demographics of New York City?

Original question: Please compare the sports environment of cities in North America.
SQL query: SELECT city_name FROM cities WHERE continent = 'North America' LIMIT 3
SQL response: The cities in North America are New York, San Francisco, and Toronto.
New question: What sports are played in New York, San Francisco, and Toronto?

Original question: What is the city with the highest population?
SQL query: SELECT city, population FROM cities ORDER BY population DESC LIMIT 1
SQL response: The city with the highest population is New York City.
New question: None

Original question: What countries are the top 3 ATP players from?
SQL query: SELECT country FROM players WHERE rank <= 3
SQL response: The top 3 ATP players are from Serbia, Russia, and Spain.
New question: None

Original question: {query_str}
SQL query: {sql_query_str}
SQL response: {sql_response_str}
New question: "
"""
DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT = PromptTemplate(
    DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT_TMPL
)


def _default_check_stop(query_bundle: QueryBundle) -> bool:
    """Default check stop function."""
    return query_bundle.query_str.lower() == "none"


def _format_sql_query(sql_query: str) -> str:
    """Format SQL query."""
    return sql_query.replace("\n", " ").replace("\t", " ")


class SQLAugmentQueryTransform(BaseQueryTransform):
    """SQL Augment Query Transform.

    This query transform will transform the query into a more specific query
    after augmenting with SQL results.

    Args:
        llm (LLM): LLM to use for query transformation.
        sql_augment_transform_prompt (BasePromptTemplate): PromptTemplate to use
            for query transformation.
        check_stop_parser (Optional[Callable[[str], bool]]): Check stop function.

    """

    def __init__(
        self,
        llm: Optional[LLMPredictorType] = None,
        sql_augment_transform_prompt: Optional[BasePromptTemplate] = None,
        check_stop_parser: Optional[Callable[[QueryBundle], bool]] = None,
    ) -> None:
        """Initialize params."""
        self._llm = llm or resolve_llm("default")

        self._sql_augment_transform_prompt = (
            sql_augment_transform_prompt or DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT
        )
        self._check_stop_parser = check_stop_parser or _default_check_stop

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"sql_augment_transform_prompt": self._sql_augment_transform_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "sql_augment_transform_prompt" in prompts:
            self._sql_augment_transform_prompt = prompts["sql_augment_transform_prompt"]

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        """Run query transform."""
        query_str = query_bundle.query_str
        sql_query = metadata["sql_query"]
        sql_query_response = metadata["sql_query_response"]
        new_query_str = self._llm.predict(
            self._sql_augment_transform_prompt,
            query_str=query_str,
            sql_query_str=sql_query,
            sql_response_str=sql_query_response,
        )
        return QueryBundle(
            new_query_str, custom_embedding_strs=query_bundle.custom_embedding_strs
        )

    def check_stop(self, query_bundle: QueryBundle) -> bool:
        """Check if query indicates stop."""
        return self._check_stop_parser(query_bundle)


class SQLJoinQueryEngine(BaseQueryEngine):
    """SQL Join Query Engine.

    This query engine can "Join" a SQL database results
    with another query engine.
    It can decide it needs to query the SQL database or the other query engine.
    If it decides to query the SQL database, it will first query the SQL database,
    whether to augment information with retrieved results from the other query engine.

    Args:
        sql_query_tool (QueryEngineTool): Query engine tool for SQL database.
            other_query_tool (QueryEngineTool): Other query engine tool.
        selector (Optional[Union[LLMSingleSelector, PydanticSingleSelector]]):
            Selector to use.
        service_context (Optional[ServiceContext]): Service context to use.
        sql_join_synthesis_prompt (Optional[BasePromptTemplate]):
            PromptTemplate to use for SQL join synthesis.
        sql_augment_query_transform (Optional[SQLAugmentQueryTransform]): Query
            transform to use for SQL augmentation.
        use_sql_join_synthesis (bool): Whether to use SQL join synthesis.
        callback_manager (Optional[CallbackManager]): Callback manager to use.
        verbose (bool): Whether to print intermediate results.

    """

    def __init__(
        self,
        sql_query_tool: QueryEngineTool,
        other_query_tool: QueryEngineTool,
        selector: Optional[Union[LLMSingleSelector, PydanticSingleSelector]] = None,
        service_context: Optional[ServiceContext] = None,
        sql_join_synthesis_prompt: Optional[BasePromptTemplate] = None,
        sql_augment_query_transform: Optional[SQLAugmentQueryTransform] = None,
        use_sql_join_synthesis: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize params."""
        super().__init__(callback_manager=callback_manager)
        # validate that the query engines are of the right type
        if not isinstance(
            sql_query_tool.query_engine,
            (BaseSQLTableQueryEngine, NLSQLTableQueryEngine),
        ):
            raise ValueError(
                "sql_query_tool.query_engine must be an instance of "
                "BaseSQLTableQueryEngine or NLSQLTableQueryEngine"
            )
        self._sql_query_tool = sql_query_tool
        self._other_query_tool = other_query_tool

        sql_query_engine = sql_query_tool.query_engine
        self._service_context = service_context or sql_query_engine.service_context

        self._selector = selector or get_selector_from_context(
            self._service_context, is_multi=False
        )
        assert isinstance(self._selector, (LLMSingleSelector, PydanticSingleSelector))

        self._sql_join_synthesis_prompt = (
            sql_join_synthesis_prompt or DEFAULT_SQL_JOIN_SYNTHESIS_PROMPT
        )
        self._sql_augment_query_transform = (
            sql_augment_query_transform
            or SQLAugmentQueryTransform(llm=self._service_context.llm)
        )
        self._use_sql_join_synthesis = use_sql_join_synthesis
        self._verbose = verbose

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

    def _query_sql_other(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query SQL database + other query engine in sequence."""
        # first query SQL database
        sql_response = self._sql_query_tool.query_engine.query(query_bundle)
        if not self._use_sql_join_synthesis:
            return sql_response

        sql_query = (
            sql_response.metadata["sql_query"] if sql_response.metadata else None
        )
        if self._verbose:
            print_text(f"SQL query: {sql_query}\n", color="yellow")
            print_text(f"SQL response: {sql_response}\n", color="yellow")

        # given SQL db, transform query into new query
        new_query = self._sql_augment_query_transform(
            query_bundle.query_str,
            metadata={
                "sql_query": _format_sql_query(sql_query),
                "sql_query_response": str(sql_response),
            },
        )

        if self._verbose:
            print_text(
                f"Transformed query given SQL response: {new_query.query_str}\n",
                color="blue",
            )
        logger.info(f"> Transformed query given SQL response: {new_query.query_str}")
        if self._sql_augment_query_transform.check_stop(new_query):
            return sql_response

        other_response = self._other_query_tool.query_engine.query(new_query)
        if self._verbose:
            print_text(f"query engine response: {other_response}\n", color="pink")
        logger.info(f"> query engine response: {other_response}")

        response_str = self._service_context.llm.predict(
            self._sql_join_synthesis_prompt,
            query_str=query_bundle.query_str,
            sql_query_str=sql_query,
            sql_response_str=str(sql_response),
            query_engine_query_str=new_query.query_str,
            query_engine_response_str=str(other_response),
        )
        if self._verbose:
            print_text(f"Final response: {response_str}\n", color="green")
        response_metadata = {
            **(sql_response.metadata or {}),
            **(other_response.metadata or {}),
        }
        source_nodes = other_response.source_nodes
        return Response(
            response_str,
            metadata=response_metadata,
            source_nodes=source_nodes,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query and get response."""
        # TODO: see if this can be consolidated with logic in RouterQueryEngine
        metadatas = [self._sql_query_tool.metadata, self._other_query_tool.metadata]
        result = self._selector.select(metadatas, query_bundle)
        # pick sql query
        if result.ind == 0:
            if self._verbose:
                print_text(f"Querying SQL database: {result.reason}\n", color="blue")
            logger.info(f"> Querying SQL database: {result.reason}")
            return self._query_sql_other(query_bundle)
        elif result.ind == 1:
            if self._verbose:
                print_text(
                    f"Querying other query engine: {result.reason}\n", color="blue"
                )
            logger.info(f"> Querying other query engine: {result.reason}")
            response = self._other_query_tool.query_engine.query(query_bundle)
            if self._verbose:
                print_text(f"Query Engine response: {response}\n", color="pink")
            return response
        else:
            raise ValueError(f"Invalid result.ind: {result.ind}")

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        # TODO: make async
        return self._query(query_bundle)
