"""SQL Vector query engine."""

from typing import Optional, cast, List, Dict, Optional, Any
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

logger = logging.getLogger(__name__)


# TODO: define query transformation


# TODO: define SQLVectorSynthesisPrompt
DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT_TMPL = (
    "The original question is given below.\n"
    "This question has been translated into a SQL query. Both the SQL "
    "query and the response are given below.\n"
    "Given the SQL response, the question has also been translated into a vector store query.\n"
    "The vector store query and response is given below.\n"
    "Given SQL query, SQL response, transformed vector store query, and vector store response, "
    "please synthesize a response to the original question.\n\n"
    "Original question: {query_str}\n"
    "SQL query: {sql_query_str}\n"
    "SQL response: {sql_response_str}\n"
    "Transformed vector store query: {vector_store_query_str}\n"
    "Vector store response: {vector_store_response_str}\n"
    "Response: "
)
DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT = Prompt(DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT_TMPL)


DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT_TMPL = (
    "The original question is given below.\n"
    "This question has been translated into a SQL query. Both the SQL "
    "query and the response are given below.\n"
    "The SQL response should provide additional context that can be used "
    "to make the question more specific. Please respond with the new question.\n"
    "Examples:\n\n"
    "Original question: Please give more details about the city with the highest population. \n"
    "SQL query: SELECT city, population FROM cities ORDER BY population DESC LIMIT 1\n"
    "SQL response: The city with the highest population is New York City.\n"
    "New question: What is the population of New York City?\n\n"
    "Original question: Please compare the sports environment of cities in North America.\n\n"
    "SQL query: SELECT city, sports FROM cities WHERE continent = 'North America'\n"
    "SQL response: The sports in North America are baseball, basketball, and football.\n"
    "New question: What sports are played in North America?\n\n"
    "Original question: {query_str}\n"
    "SQL query: {sql_query_str}\n"
    "SQL response: {sql_response_str}\n"
    "New question: "
)
DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT = Prompt(DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT_TMPL)


class SQLAugmentQueryTransform(BaseQueryTransform):
    """SQL Augment Query Transform.

    This query transform will transform the query into a more specific query
    after augmenting with SQL results.

    Args:
        llm_predictor (LLMPredictor): LLM predictor to use for query transformation.
    """

    def __init__(
        self,
        llm_predictor: Optional[BaseLLMPredictor] = None,
        sql_augment_transform_prompt: Optional[Prompt] = None,
    ) -> None:
        """Initialize params."""
        self._llm_predictor = llm_predictor or LLMPredictor()

        self._sql_augment_transform_prompt = (
            sql_augment_transform_prompt or DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT
        )

    def _run(self, query_bundle: QueryBundle, extra_info: Dict) -> QueryBundle:
        """Run query transform."""
        query_str = query_bundle.query_str
        sql_query = extra_info["sql_query"]
        sql_query_response = extra_info["sql_query_response"]
        new_query_str, _ = self._llm_predictor.predict(
            self._sql_augment_transform_prompt,
            query_str=query_str,
            sql_query_str=sql_query,
            sql_response_str=sql_query_response,
        )
        return QueryBundle(
            new_query_str, custom_embedding_strs=query_bundle.custom_embedding_strs
        )


class SQLAutoVectorQueryEngine(BaseQueryEngine):
    """SQL + Vector Index Auto Retriever Query Engine.

    This query engine can query both a SQL database
    as well as a vector database. It will first decide
    whether it needs to query the SQL database or vector store.
    If it decides to query the SQL database, it will also decide
    whether to augment information with retrieved results from the vector store.
    We use the VectorIndexAutoRetriever to retrieve results.

    """

    def __init__(
        self,
        sql_query_tool: QueryEngineTool,
        vector_query_tool: QueryEngineTool,
        selector: LLMSingleSelector,
        service_context: Optional[ServiceContext] = None,
        sql_vector_synthesis_prompt: Optional[Prompt] = None,
        sql_augment_query_transform: Optional[SQLAugmentQueryTransform] = None,
        use_sql_vector_synthesis: bool = True,
    ) -> None:
        """Initialize params."""
        # validate that the query engines are of the right type
        if not isinstance(sql_query_tool.query_engine, GPTNLStructStoreQueryEngine):
            raise ValueError(
                "sql_query_tool.query_engine must be an instance of GPTNLStructStoreQueryEngine"
            )
        if not isinstance(vector_query_tool.query_engine, RetrieverQueryEngine):
            raise ValueError(
                "vector_query_tool.query_engine must be an instance of RetrieverQueryEngine"
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
        self._selector = selector
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
        selector = selector or LLMSingleSelector()
        return cls(sql_query_tool, vector_query_tool, selector, **kwargs)

    def _query_sql(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query SQL database."""
        # first query SQL database
        response = self._sql_query_tool.query_engine.query(query_bundle)
        if not self._use_sql_vector_synthesis:
            return response

        sql_query = response.extra_info["sql_query"]

        # synthesize answer from vector db
        new_query = self._sql_augment_query_transform(query_bundle.query_str)
        vector_response = self._vector_query_tool.query_engine.query(new_query)

        response_str, _ = self._service_context.llm_predictor.predict(
            self._sql_vector_synthesis_prompt,
            query_str=query_bundle.query_str,
            sql_query_str=sql_query,
            sql_response_str=response.response,
            vector_store_query_str=new_query.query_str,
            vector_store_response_str=vector_response.response,
        )
        response_extra_info = {**response.extra_info, **vector_response.extra_info}
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
            logger.info(f"> Querying SQL database: {result.reason}")
            return self._query_sql(query_bundle)
        elif result.ind == 1:
            logger.info(f"> Querying vector database: {result.reason}")
            return self._vector_query_tool.query_engine.query(query_bundle)
        else:
            raise ValueError(f"Invalid result.ind: {result.ind}")
