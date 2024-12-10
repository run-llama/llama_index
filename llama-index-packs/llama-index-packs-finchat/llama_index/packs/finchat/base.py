"""Finance Chat LlamaPack class."""

from typing import Optional, List, Any

# The following imports have been adjusted to fix the ModuleNotFoundError
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.llms.openai import OpenAI
from llama_index.tools.finance import FinanceAgentToolSpec
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.readers.base import BaseReader
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.schema import Document
from llama_index.core.base.llms.types import ChatMessage
from sqlalchemy import MetaData, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.schema import CreateTable


class SQLDatabaseToolSpec(BaseToolSpec, BaseReader):
    """
    A tool to query and retrieve results from a SQL Database.

    Args:
        sql_database (Optional[SQLDatabase]): SQL database to use,
            including table names to specify.
            See :ref:`Ref-Struct-Store` for more details.

        OR

        engine (Optional[Engine]): SQLAlchemy Engine object of the database connection.

        OR

        uri (Optional[str]): uri of the database connection.

        OR

        scheme (Optional[str]): scheme of the database connection.
        host (Optional[str]): host of the database connection.
        port (Optional[int]): port of the database connection.
        user (Optional[str]): user of the database connection.
        password (Optional[str]): password of the database connection.
        dbname (Optional[str]): dbname of the database connection.

    """

    spec_functions = ["run_sql_query", "describe_tables", "list_tables"]

    def __init__(
        self,
        sql_database: Optional[SQLDatabase] = None,
        engine: Optional[Engine] = None,
        uri: Optional[str] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dbname: Optional[str] = None,
        *args: Optional[Any],
        **kwargs: Optional[Any],
    ) -> None:
        """Initialize with parameters."""
        if sql_database:
            self.sql_database = sql_database
        elif engine:
            self.sql_database = SQLDatabase(engine, *args, **kwargs)
        elif uri:
            self.uri = uri
            self.sql_database = SQLDatabase.from_uri(uri, *args, **kwargs)
        elif scheme and host and port and user and password and dbname:
            uri = f"{scheme}://{user}:{password}@{host}:{port}/{dbname}"
            self.uri = uri
            self.sql_database = SQLDatabase.from_uri(uri, *args, **kwargs)
        else:
            raise ValueError(
                "You must provide either a SQLDatabase, "
                "a SQL Alchemy Engine, a valid connection URI, or a valid "
                "set of credentials."
            )
        self._metadata = MetaData()
        self._metadata.reflect(bind=self.sql_database.engine)

    def run_sql_query(self, query: str) -> Document:
        r"""Runs SQL query on the provided SQL database, returning a Document storing all the rows separated by \n.

        Args:
            query (str): SQL query in text format which can directly be executed using SQLAlchemy engine.

        Returns:
            Document: Document storing all the output result of the sql-query generated.
        """
        with self.sql_database.engine.connect() as connection:
            if query is None:
                raise ValueError("A query parameter is necessary to filter the data")
            else:
                result = connection.execute(text(query))
            all_doc_str = ""
            for item in result.fetchall():
                if all_doc_str:
                    all_doc_str += "\n"
                # fetch each item
                doc_str = ", ".join([str(entry) for entry in item])
                all_doc_str += doc_str
            return Document(text=all_doc_str)

    def list_tables(self) -> List[str]:
        """
        Returns a list of available tables in the database.
        """
        return [x.name for x in self._metadata.sorted_tables]

    def describe_tables(self, tables: Optional[List[str]] = None) -> str:
        """
        Describes the specified tables in the database.

        Args:
            tables (List[str]): A list of table names to retrieve details about
        """
        table_names = tables or [table.name for table in self._metadata.sorted_tables]
        table_schemas = []

        for table_name in table_names:
            table = next(
                (
                    table
                    for table in self._metadata.sorted_tables
                    if table.name == table_name
                ),
                None,
            )
            if table is None:
                raise NoSuchTableError(f"Table '{table_name}' does not exist.")
            schema = str(CreateTable(table).compile(self.sql_database._engine))
            table_schemas.append(f"{schema}\n")

        return "\n".join(table_schemas)

    def get_table_info(self) -> str:
        """Construct table info for the all tables in DB which includes information about the columns of the table and also shows top row of the table."""
        all_table_info = ""
        for table_name in self.list_tables():
            table_info = self.sql_database.get_single_table_info(table_name)
            table_info += f"\n\nHere is the DDL statement for this table:\n"
            table_info += self.describe_tables([table_name])
            _, output = self.sql_database.run_sql(f"SELECT * FROM {table_name} LIMIT 1")
            table_info += f"\nTop row of {table_name}:\n\n"
            for colname in output["col_keys"]:
                table_info += colname + "\t"
            table_info += "\n"
            for data in output["result"]:
                for val in data:
                    table_info += str(val) + "\t"
                table_info += "\n"
            all_table_info += f"\n{table_info}\n"
        return all_table_info


class FinanceChatPack(BaseLlamaPack):
    def __init__(
        self,
        polygon_api_key: str,
        finnhub_api_key: str,
        alpha_vantage_api_key: str,
        newsapi_api_key: str,
        openai_api_key: str,
        postgres_db_uri: str,
        gpt_model_name: str = "gpt-4-0613",
    ):
        llm = OpenAI(temperature=0, model=gpt_model_name, api_key=openai_api_key)
        self.db_tool_spec = SQLDatabaseToolSpec(uri=postgres_db_uri)
        self.fin_tool_spec = FinanceAgentToolSpec(
            polygon_api_key, finnhub_api_key, alpha_vantage_api_key, newsapi_api_key
        )

        self.db_table_info = self.db_tool_spec.get_table_info()
        prefix_messages = self.construct_prefix_db_message(self.db_table_info)
        # add some role play in the system .
        database_agent = OpenAIAgent.from_tools(
            [
                tool
                for tool in self.db_tool_spec.to_tool_list()
                if tool.metadata.name == "run_sql_query"
            ],
            prefix_messages=prefix_messages,
            llm=llm,
            verbose=True,
        )
        database_agent_tool = QueryEngineTool.from_defaults(
            database_agent,
            name="database_agent",
            description=""""
                This agent analyzes a text query and add further explanations and thoughts to help a data scientist who has access to following tables:

                {table_info}

                Be concise and do not lose any information about original query while passing to the data scientist.
                """,
        )

        fin_api_agent = OpenAIAgent.from_tools(
            self.fin_tool_spec.to_tool_list(),
            system_prompt=f"""
                You are a helpful AI financial assistant designed to understand the intent of the user query and then use relevant tools/apis to help answer it.
                You can use more than one tool/api only if needed, but final response should be concise and relevant. If you are not able to find
                relevant tool/api, respond respectfully suggesting that you don't know. Think step by step""",
            llm=llm,
            verbose=True,
        )

        fin_api_agent_tool = QueryEngineTool.from_defaults(
            fin_api_agent,
            name="fin_api_agent",
            description=f"""
                This agent has access to another agent which can access certain open APIs to provide information based on user query.
                Analyze the query and add any information if needed which can help to decide which API to call.
                Be concise and do not lose any information about original query.
                """,
        )

        self.fin_hierarchical_agent = OpenAIAgent.from_tools(
            [database_agent_tool, fin_api_agent_tool],
            system_prompt="""
                You are a specialized financial assistant with access to certain tools which can access open APIs and SP500 companies database containing information on
                daily opening price, closing price, high, low, volume, reported earnings, estimated earnings since 2010 to 2023. Before answering query you should check
                if the question can be answered via querying the database or using specific open APIs. If you try to find answer via querying database first and it did
                not work out, think if you can use other tool APIs available before replying gracefully.
                """,
            llm=llm,
            verbose=True,
        )

    def construct_prefix_db_message(self, table_info: str) -> str:
        system_prompt = f"""
        You are a smart data scientist working in a reputed trading firm like Jump Trading developing automated trading algorithms. Take a deep breathe and think
        step by step to design queries over a SQL database.

        Here is a complete description of tables in SQL database you have access to:

        {table_info}

        Use responses to past questions also to guide you.


        """

        prefix_messages = []
        prefix_messages.append(ChatMessage(role="system", content=system_prompt))

        prefix_messages.append(
            ChatMessage(
                role="user",
                content="What is the average price of Google in the month of July in 2023",
            )
        )
        prefix_messages.append(
            ChatMessage(
                role="assistant",
                content="""
        SELECT AVG(close) AS AvgPrice
        FROM stock_data
        WHERE stock = 'GOOG'
            AND date >= '2023-07-01'
            AND date <= '2023-07-31';
        """,
            )
        )

        prefix_messages.append(
            ChatMessage(
                role="user",
                content="Which stock has the maximum % change in any month in 2023",
            )
        )
        # prefix_messages.append(ChatMessage(role="user", content="Which stocks gave more than 2% return constantly in month of July from past 5 years"))
        prefix_messages.append(
            ChatMessage(
                role="assistant",
                content="""
        WITH MonthlyPrices AS (
            SELECT
                stock,
                EXTRACT(YEAR FROM date) AS year,
                EXTRACT(MONTH FROM date) AS month,
                FIRST_VALUE(close) OVER (PARTITION BY stock, EXTRACT(YEAR FROM date), EXTRACT(MONTH FROM date) ORDER BY date ASC) AS opening_price,
                LAST_VALUE(close) OVER (PARTITION BY stock, EXTRACT(YEAR FROM date), EXTRACT(MONTH FROM date) ORDER BY date ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS closing_price
            FROM
                stock_data
            WHERE
                EXTRACT(YEAR FROM date) = 2023
        ),
        PercentageChanges AS (
            SELECT
                stock,
                year,
                month,
                CASE
                    WHEN opening_rice IS NULL OR closing_price IS NULL THEN NULL
                    WHEN opening_price = 0 THEN NULL
                    ELSE ((closing_price - opening_price) / opening_price) * 100
                END AS pct
            FROM
                MonthlyPrices
        )
        SELECT *
        FROM
            PercentageChanges
        WHERE pct IS NOT NULL
        ORDER BY
            pct DESC
        LIMIT 1;
        """,
            )
        )

        prefix_messages.append(
            ChatMessage(
                role="user",
                content="How many times Microsoft beat earnings estimates in 2022",
            )
        )
        prefix_messages.append(
            ChatMessage(
                role="assistant",
                content="""
        SELECT
            COUNT(*)
        FROM
            earnings
        WHERE
            stock = 'MSFT' AND reported > estimated and EXTRACT(YEAR FROM date) = 2022
        """,
            )
        )

        prefix_messages.append(
            ChatMessage(
                role="user",
                content="Which stocks have beaten earnings estimate by more than 1$ consecutively from last 4 reportings?",
            )
        )
        prefix_messages.append(
            ChatMessage(
                role="assistant",
                content="""
        WITH RankedEarnings AS(
            SELECT
                stock,
                date,
                reported,
                estimated,
                RANK() OVER (PARTITION BY stock ORDER BY date DESC) as ranking
            FROM
                earnings
        )
        SELECT
            stock
        FROM
            RankedEarnings
        WHERE
            ranking <= 4 AND reported - estimated > 1
        GROUP BY
            stock
        HAVING COUNT(*) = 4
        """,
            )
        )

        return prefix_messages

    def run(self, query: str):
        return self.fin_hierarchical_agent.chat(query)
