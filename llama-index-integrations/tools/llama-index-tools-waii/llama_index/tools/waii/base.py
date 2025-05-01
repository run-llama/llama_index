"""Waii Tool."""

import json
from typing import Any, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class WaiiToolSpec(BaseToolSpec, BaseReader):
    spec_functions = [
        "get_answer",
        "describe_query",
        "performance_analyze",
        "diff_query",
        "describe_dataset",
        "transcode",
        "get_semantic_contexts",
        "generate_query_only",
        "run_query",
    ]

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        database_key: Optional[str] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        from waii_sdk_py import WAII

        WAII.initialize(url=url, api_key=api_key)
        WAII.Database.activate_connection(key=database_key)
        self.verbose = verbose

    def _try_display(self, obj) -> None:
        # only display when verbose is True, we don't want to display too much information by default.
        if self.verbose:
            try:
                from IPython.display import display

                # display df if the function `display` is available (display only available when running with IPYTHON),
                # if it is not available, just ignore the exception.
                display(obj)
            except ImportError:
                # Handle the case where IPython is not available.
                pass

    def _run_query(self, sql: str, return_summary: bool) -> List[Document]:
        from waii_sdk_py import WAII
        from waii_sdk_py.query import RunQueryRequest

        run_result = WAII.Query.run(RunQueryRequest(query=sql))

        self._try_display(run_result.to_pandas_df())

        # create documents based on returned rows
        documents = [Document(text=str(doc)) for doc in run_result.rows]

        if return_summary:
            return self._get_summarization(
                "Summarize the result in text, don't miss any detail.", documents
            )

        return documents

    def load_data(self, ask: str) -> List[Document]:
        """
        Query using natural language and load data from the Database, returning a list of Documents.

        Args:
            ask: a natural language question.

        Returns:
            List[Document]: A list of Document objects.

        """
        query = self.generate_query_only(ask)

        return self._run_query(query, False)

    def _get_summarization(self, original_ask: str, documents) -> Any:
        texts = []

        n_chars = 0
        for i in range(len(documents)):
            t = str(documents[i].text)
            if len(t) + n_chars > 8192:
                texts.append(f"... {len(documents) - i} more results")
                break
            texts.append(t)
            n_chars += len(t)

        summarizer = TreeSummarize()
        return summarizer.get_response(original_ask, texts)

    def get_answer(self, ask: str) -> List[Document]:
        """
        Generate a SQL query and run it against the database, returning the summarization of the answer
        Args:
            ask: a natural language question.

        Returns:
            str: A string containing the summarization of the answer.

        """
        query = self.generate_query_only(ask)

        return self._run_query(query, True)

    def generate_query_only(self, ask: str) -> str:
        """
        Generate a SQL query and NOT run it, returning the query. If you need to get answer, you should use get_answer instead.

        Args:
            ask: a natural language question.

        Returns:
            str: A string containing the query.

        """
        from waii_sdk_py import WAII
        from waii_sdk_py.query import QueryGenerationRequest

        query = WAII.Query.generate(QueryGenerationRequest(ask=ask)).query

        self._try_display(query)

        return query

    def run_query(self, sql: str) -> List[Document]:
        return self._run_query(sql, False)

    def describe_query(self, question: str, query: str) -> str:
        """
        Describe a sql query, returning the summarization of the answer.

        Args:
            question: a natural language question which the people want to ask.
            query: a sql query.

        Returns:
            str: A string containing the summarization of the answer.

        """
        from waii_sdk_py import WAII
        from waii_sdk_py.query import DescribeQueryRequest

        result = WAII.Query.describe(DescribeQueryRequest(query=query))
        result = json.dumps(result.dict(), indent=2)
        self._try_display(result)

        return self._get_summarization(question, [Document(text=result)])

    def performance_analyze(self, query_uuid: str) -> str:
        """
        Analyze the performance of a query, returning the summarization of the answer.

        Args:
            query_uuid: a query uuid, e.g. xxxxxxxxxxxxx...

        Returns:
            str: A string containing the summarization of the answer.

        """
        from waii_sdk_py import WAII
        from waii_sdk_py.query import QueryPerformanceRequest

        result = WAII.Query.analyze_performance(
            QueryPerformanceRequest(query_id=query_uuid)
        )
        return json.dumps(result.dict(), indent=2)

    def diff_query(self, previous_query: str, current_query: str) -> str:
        """
        Diff two sql queries, returning the summarization of the answer.

        Args:
            previous_query: previous sql query.
            current_query: current sql query.

        Returns:
            str: A string containing the summarization of the answer.

        """
        from waii_sdk_py import WAII
        from waii_sdk_py.query import DiffQueryRequest

        result = WAII.Query.diff(
            DiffQueryRequest(query=current_query, previous_query=previous_query)
        )
        result = json.dumps(result.dict(), indent=2)
        return self._get_summarization("get diff summary", [Document(text=result)])

    def describe_dataset(
        self,
        ask: str,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> str:
        """
        Describe a dataset (no matter if it is a table or schema), returning the summarization of the answer.
        Example questions like: "describe the dataset", "what the schema is about", "example question for the table xxx", etc.
        When both schema and table are None, describe the whole database.

        Args:
            ask: a natural language question (how you want to describe the dataset).
            schema_name: a schema name (shouldn't include the database name or the table name).
            table_name: a table name. (shouldn't include the database name or the schema name).

        Returns:
            str: A string containing the summarization of the answer.

        """
        from waii_sdk_py import WAII

        catalog = WAII.Database.get_catalogs()

        # filter by schema / table
        schemas = {}
        tables = {}

        for c in catalog.catalogs:
            for s in c.schemas:
                for t in s.tables:
                    if (
                        schema_name is not None
                        and schema_name.lower() != t.name.schema_name.lower()
                    ):
                        continue
                    if table_name is not None:
                        if table_name.lower() != t.name.table_name.lower():
                            continue
                        tables[str(t.name)] = t
                    schemas[str(s.name)] = s

        # remove tables ref from schemas
        for schema in schemas:
            schemas[schema].tables = None

        # generate response
        return self._get_summarization(
            ask + ", use the provided information to get comprehensive summarization",
            [Document(text=str(schemas[schema])) for schema in schemas]
            + [Document(text=str(tables[table])) for table in tables],
        )

    def transcode(
        self,
        instruction: Optional[str] = "",
        source_dialect: Optional[str] = None,
        source_query: Optional[str] = None,
        target_dialect: Optional[str] = None,
    ) -> str:
        """
        Transcode a sql query from one dialect to another, returning generated query.

        Args:
            instruction: instruction in natural language.
            source_dialect: the source dialect of the query.
            source_query: the source query.
            target_dialect: the target dialect of the query.

        Returns:
            str: A string containing the generated query.

        """
        from waii_sdk_py import WAII
        from waii_sdk_py.query import TranscodeQueryRequest

        result = WAII.Query.transcode(
            TranscodeQueryRequest(
                ask=instruction,
                source_dialect=source_dialect,
                source_query=source_query,
                target_dialect=target_dialect,
            )
        )
        return result.query

    def get_semantic_contexts(self) -> Any:
        """Get all pre-defined semantic contexts."""
        from waii_sdk_py import WAII

        return WAII.SemanticContext.get_semantic_context().semantic_context
