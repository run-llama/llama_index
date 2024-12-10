"""Vanna AI Pack.

Uses: https://vanna.ai/.

"""

from typing import Any, Dict, Optional, cast

from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.query_engine import CustomQueryEngine
import pandas as pd
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response


class VannaQueryEngine(CustomQueryEngine):
    """Vanna query engine.

    Uses chromadb and OpenAI.

    """

    openai_api_key: str
    sql_db_url: str

    ask_kwargs: Dict[str, Any]
    vn: Any

    def __init__(
        self,
        openai_api_key: str,
        sql_db_url: str,
        openai_model: str = "gpt-3.5-turbo",
        ask_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        from vanna.openai.openai_chat import OpenAI_Chat
        from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

        class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
            def __init__(self, config: Any = None) -> None:
                ChromaDB_VectorStore.__init__(self, config=config)
                OpenAI_Chat.__init__(self, config=config)

        vn = MyVanna(config={"api_key": openai_api_key, "model": openai_model})
        vn.connect_to_sqlite(sql_db_url)
        if verbose:
            print(f"> Connected to db: {sql_db_url}")

        # get every table DDL from db
        sql_results = cast(
            pd.DataFrame,
            vn.run_sql("SELECT sql FROM sqlite_master WHERE type='table';"),
        )
        # go through every sql statement, do vn.train(ddl=ddl) on it
        for idx, sql_row in sql_results.iterrows():
            if verbose:
                print(f"> Training on {sql_row['sql']}")
            vn.train(ddl=sql_row["sql"])

        super().__init__(
            openai_api_key=openai_api_key,
            sql_db_url=sql_db_url,
            vn=vn,
            ask_kwargs=ask_kwargs or {},
            **kwargs,
        )

    def custom_query(self, query_str: str) -> RESPONSE_TYPE:
        """Query."""
        from vanna.base import VannaBase

        vn = cast(VannaBase, self.vn)
        ask_kwargs = {"visualize": False, "print_results": False}
        ask_kwargs.update(self.ask_kwargs)
        sql = vn.generate_sql(
            query_str,
            **ask_kwargs,
        )
        result = vn.run_sql(sql)
        if result is None:
            raise ValueError("Vanna returned None.")
        sql, df, _ = result

        return Response(response=str(df), metadata={"sql": sql, "df": df})


class VannaPack(BaseLlamaPack):
    """Vanna AI pack.

    Uses OpenAI and ChromaDB. Of course Vanna.AI allows you to connect to many more dbs
    and use more models - feel free to refer to their page for more details:
    https://vanna.ai/docs/snowflake-openai-vanna-vannadb.html

    """

    def __init__(
        self,
        openai_api_key: str,
        sql_db_url: str,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.vanna_query_engine = VannaQueryEngine(
            openai_api_key=openai_api_key, sql_db_url=sql_db_url, **kwargs
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "vanna_query_engine": self.vanna_query_engine,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.vanna_query_engine.query(*args, **kwargs)
