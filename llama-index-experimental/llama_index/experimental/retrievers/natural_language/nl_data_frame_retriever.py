from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.llms import llm
from llama_index.core.llms.utils import resolve_llm
from llama_index.core.utils import print_text
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from typing import List, Optional

import asyncio
import logging
import pandas as pd
import duckdb
import re

logger = logging.getLogger(__name__)

DEFAULT_OWL_GENERATOR_PROMPT_TMPL = """\
given the schema:{schema}\
generate the owl representation of the schema.\
make sure to take into account types as described in schema.org\
Each row would be an instance of the schema and each column would be a property of the schema.\
Some properties could be entities and others could be literals.\
For entities, make sure to use or extend the schema.org vocabulary.\
When are group of columns are related to each other, make sure to use the schema.org vocabulary to represent the relationship.\
Soe column group might identifie other entities, some might be a description of the entity, some might be a location, etc.\
Try to use the most specific schema.org vocabulary possible.\
Do not introduce new labels when possible, for example, if a column is a date, use the schema.org vocabulary for date.\
If a column is called first name, use the schema.org vocabulary for givenName.\

if the table name is table_data or it is too generic do not name the schema after the table name, instead name it after the domain of the table.\

Emit the owl representation of the schema only.\
"""

DEFAULT_OWL_GENERATOR_PROMPT = PromptTemplate(
    DEFAULT_OWL_GENERATOR_PROMPT_TMPL,
    prompt_type=PromptType.CUSTOM,
)

DEFAULT_USE_DECTECTION_TMPL = """\
given the schema:{schema}\
describe what this retriever is useful for. What kind of information  can the retriever provide and the type of data it can access.\
"""

DEFAULT_USE_DECTECTION_PROMPT = PromptTemplate(
    DEFAULT_USE_DECTECTION_TMPL,
    prompt_type=PromptType.CUSTOM,
)

DEFAULT_RESULT_RANKING_TMPL = """\
given the schema:{schema}\
and the query: {query}\
how relevant is the schema?\
the relevance must be a number between 0 and 1 where 1 indicates that the schema is able to model the domain of the query and 0 indicates that the schema is not able to model the domain of the query.\

produce only the numeric value and nothing else.\
relevance:
"""

DEFAULT_RESULT_RANKING_PROMPTROMPT = PromptTemplate(
    DEFAULT_RESULT_RANKING_TMPL,
    prompt_type=PromptType.CUSTOM,
)


class NLDataframeRetriever(BaseRetriever):
    def __init__(
        self,
        df: pd.DataFrame,
        llm: llm,
        name: Optional[str] = None,
        text_to_sql_prompt: Optional[BasePromptTemplate] = None,
        schema_to_owl_prompt: Optional[BasePromptTemplate] = None,
        schema_use_detection_prompt: Optional[BasePromptTemplate] = None,
        result_ranking_prompt: Optional[BasePromptTemplate] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ):
        self._llm = resolve_llm(llm)
        self._similarity_top_k = similarity_top_k
        self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        self._result_ranking_prompt = (
            result_ranking_prompt or DEFAULT_RESULT_RANKING_PROMPTROMPT
        )
        self._schema_to_owl_prompt = (
            schema_to_owl_prompt or DEFAULT_OWL_GENERATOR_PROMPT
        )
        self._schema_use_detection_prompt = (
            schema_use_detection_prompt or DEFAULT_USE_DECTECTION_PROMPT
        )
        data_source = df.copy(deep=True)
        data_source.rename(columns=lambda x: re.sub(r"\s+", "_", x), inplace=True)
        table_name = name or "data_table"
        self._connection = duckdb.connect()
        self._connection.sql(f"CREATE TABLE {table_name} AS SELECT * FROM data_source")
        self._connection.sql(
            f"INSERT INTO {table_name} BY NAME SELECT * FROM data_source"
        )
        self._schema = self._connection.table(f"{table_name}").describe()
        self._description = None
        self._owl = None
        self._schema_str = self._create_schema()

        super().__init__(callback_manager=callback_manager, verbose=verbose)

    def get_owl(self) -> str:
        if self._owl is None:
            response = self._llm.complete(
                self._schema_to_owl_prompt, schema=self._schema_str
            )

            logger.info(f"Schema Description: {response.text}")

            self._owl = response.text
        return self._owl

    def get_description(self) -> str:
        if self._description is None:
            response = self._llm.complete(
                self._schema_use_detection_prompt, schema=self._schema_str
            )

            logger.info(f"Schema Description: {response.text}")

            self._description = response.text

        return self._description

    def _create_schema(self) -> str:
        table_desc_list = []
        table_desc_list.append(f"{self._schema.alias} (")
        for idx, column in enumerate(self._schema.columns):
            table_desc_list.append(f"  {column} {self._schema.types[idx]},")
        table_desc_list.append(f")")

        tables_desc_str = "\n\n".join(table_desc_list)
        logger.info(f"Schema: {tables_desc_str}")

        return tables_desc_str

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        tables_desc_str = self._schema_str

        if self._verbose:
            print_text(
                f"Executing retrieval for {query_bundle.query_str} using Schema: {tables_desc_str}\n",
                color="llama_pink",
            )

        response_str = self._llm.predict(
            self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=tables_desc_str,
            dialect="PL/pgSQL",
        )

        rank = self._llm.complete(
            self._result_ranking_prompt,
            query=query_bundle.query_str,
            schema=tables_desc_str,
        )

        score = 1.0
        try:
            score = float(rank.text)
        except ValueError as parsing_error:
            logger.error(
                f"Error in parsing the rank value: {rank.text} : {parsing_error}"
            )

        query = self._parse_response_to_sql(response_str)
        if self._verbose:
            print_text(
                f"Executing query: {query}\n",
                color="llama_turquoise",
            )

        logger.info(f"Executing query: {query}")

        results = []
        if self._similarity_top_k <= 0:
            results = self._connection.execute(query).fetchall()
        else:
            cursor = self._connection.execute(query)

            for i in range(self._similarity_top_k):
                result = cursor.fetchone()
                if result is None:
                    break
                results.append(result)

            cursor.fetchall()

        retrived_nodes: List[NodeWithScore] = []
        for result in results:
            retrived_nodes.append(
                NodeWithScore(node=TextNode(text=f"{result}"), score=score)
            )

        return retrived_nodes

    def _parse_response_to_sql(self, response: str) -> str:
        """Parse response to SQL."""
        # Find and remove SQLResult part
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]
        return response.strip()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return asyncio.run(self._aretrieve(query_bundle))
