"""SQL Vector query engine."""

from langchain.input import print_text
from typing import Optional, cast, Dict, Any, Callable
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.struct_store.sql_query import GPTNLStructStoreQueryEngine
from llama_index.indices.vector_store.retrievers.auto_retriever import (
    VectorIndexAutoRetriever,
)
from llama_index.indices.query.schema import QueryBundle
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.selectors.llm_selectors import LLMSingleSelector
from llama_index.prompts.base import Prompt
from llama_index.indices.query.query_transform.base import BaseQueryTransform
import logging
from llama_index.langchain_helpers.chain_wrapper import LLMPredictor
from llama_index.llm_predictor.base import BaseLLMPredictor
from llama_index.callbacks.base import CallbackManager

logger = logging.getLogger(__name__)


DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT_TMPL = """
The original question is given below.
This question has been translated into a SQL query. Both the SQL query and the response are given below.
Given the SQL response, the question has also been translated into a vector store query.
The vector store query and response is given below.
Given SQL query, SQL response, transformed vector store query, and vector store response, please synthesize a response to the original question.

Original question: {query_str}
SQL query: {sql_query_str}
SQL response: {sql_response_str}
Transformed vector store query: {vector_store_query_str}
Vector store response: {vector_store_response_str}
Response: 
"""  # noqa
DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT = Prompt(DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT_TMPL)


DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT_TMPL = """
"The original question is given below.
This question has been translated into a SQL query. Both the SQL query and the response are given below.
The SQL response either answers the question, or should provide additional context that can be used to make the question more specific.
Your job is to come up with a more specific question that needs to be answered to fully answer the original question, or 'None' if the original question has already been fully answered from the SQL response. Do not create a new question that is irrelevant to the original question; in that case return None instead.

Examples:

Original question: Please give more details about the demographics of the city with the highest population.
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
"""  # noqa
DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT = Prompt(DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT_TMPL)


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
        llm_predictor (LLMPredictor): LLM predictor to use for query transformation.
        sql_augment_transform_prompt (Prompt): Prompt to use for query transformation.
        check_stop_parser (Optional[Callable[[str], bool]]): Check stop function.

    """

    def __init__(
        self,
        llm_predictor: Optional[BaseLLMPredictor] = None,
        sql_augment_transform_prompt: Optional[Prompt] = None,
        check_stop_parser: Optional[Callable[[QueryBundle], bool]] = None,
    ) -> None:
        """Initialize params."""
        self._llm_predictor = llm_predictor or LLMPredictor()

        self._sql_augment_transform_prompt = (
            sql_augment_transform_prompt or DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT
        )
        self._check_stop_parser = check_stop_parser or _default_check_stop

    def _run(self, query_bundle: QueryBundle, extra_info: Dict) -> QueryBundle:
        """Run query transform."""
        query_str = query_bundle.query_str
        sql_query = extra_info["sql_query"]
        sql_query_response = extra_info["sql_query_response"]
        new_query_str, formatted_prompt = self._llm_predictor.predict(
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


class SQLAutoVectorQueryEngine(BaseQueryEngine):
    """SQL + Vector Index Auto Retriever Query Engine.

    This query engine can query both a SQL database
    as well as a vector database. It will first decide
    whether it needs to query the SQL database or vector store.
    If it decides to query the SQL database, it will also decide
    whether to augment information with retrieved results from the vector store.
    We use the VectorIndexAutoRetriever to retrieve results.

    Args:
        sql_query_tool (QueryEngineTool): Query engine tool for SQL database.
        vector_query_tool (QueryEngineTool): Query engine tool for vector database.
        selector (Optional[LLMSingleSelector]): Selector to use.
        service_context (Optional[ServiceContext]): Service context to use.
        sql_vector_synthesis_prompt (Optional[Prompt]): Prompt to use for SQL vector
            synthesis.
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
        selector: Optional[LLMSingleSelector] = None,
        service_context: Optional[ServiceContext] = None,
        sql_vector_synthesis_prompt: Optional[Prompt] = None,
        sql_augment_query_transform: Optional[SQLAugmentQueryTransform] = None,
        use_sql_vector_synthesis: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize params."""
        super().__init__(callback_manager=callback_manager)
        # validate that the query engines are of the right type
        if not isinstance(sql_query_tool.query_engine, GPTNLStructStoreQueryEngine):
            raise ValueError(
                "sql_query_tool.query_engine must be an instance of "
                "GPTNLStructStoreQueryEngine"
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
        self._sql_query_tool = sql_query_tool
        self._vector_query_tool = vector_query_tool

        sql_query_engine = cast(
            GPTNLStructStoreQueryEngine, sql_query_tool.query_engine
        )
        self._service_context = service_context or sql_query_engine.service_context
        self._selector = selector or LLMSingleSelector.from_defaults()
        self._sql_vector_synthesis_prompt = (
            sql_vector_synthesis_prompt or DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT
        )
        self._sql_augment_query_transform = (
            sql_augment_query_transform
            or SQLAugmentQueryTransform(
                llm_predictor=self._service_context.llm_predictor
            )
        )
        self._use_sql_vector_synthesis = use_sql_vector_synthesis
        self._verbose = verbose

    @classmethod
    def from_sql_and_vector_query_engines(
        cls,
        sql_query_engine: GPTNLStructStoreQueryEngine,
        sql_tool_name: str,
        sql_tool_description: str,
        vector_auto_retriever: RetrieverQueryEngine,
        vector_tool_name: str,
        vector_tool_description: str,
        selector: Optional[LLMSingleSelector] = None,
        **kwargs: Any,
    ) -> "SQLAutoVectorQueryEngine":
        """From SQL and vector query engines.

        Args:
            sql_query_engine (GPTNLStructStoreQueryEngine): SQL query engine.
            vector_query_engine (VectorIndexAutoRetriever): Vector retriever.
            selector (Optional[LLMSingleSelector]): Selector to use.

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

    def _query_sql_vector(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query SQL database + vector db in sequence."""
        # first query SQL database
        sql_response = self._sql_query_tool.query_engine.query(query_bundle)
        if not self._use_sql_vector_synthesis:
            return sql_response

        sql_query = (
            sql_response.extra_info["sql_query"] if sql_response.extra_info else None
        )
        if self._verbose:
            print_text(f"SQL query: {sql_query}\n", color="yellow")
            print_text(f"SQL response: {sql_response}\n", color="yellow")

        # given SQL db, transform query into new query
        new_query = self._sql_augment_query_transform(
            query_bundle.query_str,
            extra_info={
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

        vector_response = self._vector_query_tool.query_engine.query(new_query)
        if self._verbose:
            print_text(f"Vector DB response: {vector_response}\n", color="pink")
        logger.info(f"> Vector DB response: {vector_response}")

        response_str, _ = self._service_context.llm_predictor.predict(
            self._sql_vector_synthesis_prompt,
            query_str=query_bundle.query_str,
            sql_query_str=sql_query,
            sql_response_str=str(sql_response),
            vector_store_query_str=new_query.query_str,
            vector_store_response_str=str(vector_response),
        )
        if self._verbose:
            print_text(f"Final response: {response_str}\n", color="green")
        response_extra_info = {
            **(sql_response.extra_info or {}),
            **(vector_response.extra_info or {}),
        }
        source_nodes = vector_response.source_nodes
        return Response(
            response_str,
            extra_info=response_extra_info,
            source_nodes=source_nodes,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query and get response."""
        # TODO: see if this can be consolidated with logic in RouterQueryEngine
        metadatas = [self._sql_query_tool.metadata, self._vector_query_tool.metadata]
        result = self._selector.select(metadatas, query_bundle)
        # pick sql query
        if result.ind == 0:
            if self._verbose:
                print_text(f"Querying SQL database: {result.reason}\n", color="blue")
            logger.info(f"> Querying SQL database: {result.reason}")
            return self._query_sql_vector(query_bundle)
        elif result.ind == 1:
            if self._verbose:
                print_text(f"Querying vector database: {result.reason}\n", color="blue")
            logger.info(f"> Querying vector database: {result.reason}")
            response = self._vector_query_tool.query_engine.query(query_bundle)
            if self._verbose:
                print_text(f"Vector DB response: {response}\n", color="pink")
            return response
        else:
            raise ValueError(f"Invalid result.ind: {result.ind}")

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        # TODO: make async
        return self._query(query_bundle)
