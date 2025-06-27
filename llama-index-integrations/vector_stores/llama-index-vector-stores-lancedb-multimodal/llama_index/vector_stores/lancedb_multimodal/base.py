import lancedb
import os

from lancedb.pydantic import LanceModel
from lancedb import DBConnection
from lancedb.table import Table
from lancedb.types import IndexType
from lancedb.rerankers import Reranker

from typing import Optional, Dict, Any
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.bridge.pydantic import PrivateAttr

DEFAULT_TABLE_NAME = "default_table"


class LanceMultiModalStore(BasePydanticVectorStore):
    """
    Implementation of the MultiModal AI LakeHouse by LanceDB.
    """

    class Config:
        arbitrary_types_allowed = True

    connection: Optional[DBConnection]
    uri: Optional[str]
    api_key: Optional[str]
    region: Optional[str]
    table_name: Optional[str]
    table_schema: Optional[LanceModel]
    infer_table_schema: bool
    embedding_model: Optional[str]
    table_exists: bool
    create_index: bool
    indexing_strategy: IndexType
    reranker: Optional[Reranker]

    _connection: DBConnection = PrivateAttr(
        default=None,
    )
    _table: Table = PrivateAttr(default=None)
    _reranker: Optional[Reranker] = PrivateAttr(default=None)

    def __init__(
        self,
        connection: Optional[DBConnection] = None,
        uri: Optional[str] = None,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        table_name: Optional[str] = None,
        table_schema: Optional[LanceModel] = None,
        table_exists: bool = False,
        create_index: bool = True,
        indexing_strategy: IndexType = "IVF_PQ",
        indexing_kwargs: Dict[str, Any] = {},
        reranker: Optional[Reranker] = None,
    ) -> None:
        self._reranker = reranker
        if connection:
            self._connection = connection
        else:
            if uri and not uri.startswith("db://"):
                self._connection = lancedb.connect(uri)
            elif uri and uri.startswith("db://") and not api_key:
                api_key = os.getenv("LANCEDB_API_KEY", None)
                if not api_key:
                    raise ValueError(
                        "LanceDB API key is neither provide nor set as an env variable."
                    )
                else:
                    if not region:
                        region = "us-east-1"
                    self._connection = lancedb.connect(
                        api_key=api_key, region=region, uri=uri
                    )
            elif uri and uri.startswith("db://") and api_key:
                if not region:
                    region = "us-east-1"
                self._connection = lancedb.connect(
                    api_key=api_key, region=region, uri=uri
                )
            else:
                raise ValueError(
                    "LanceDB connection object not provided and LanceDB URI missing or incorrect."
                )
        if table_exists:
            if not table_name:
                raise ValueError(
                    "Table name must be passed if table_exists is set to True"
                )
            self._table = self._connection.open_table(table_name)
        else:
            table_name = table_name or DEFAULT_TABLE_NAME
            if not table_schema:
                raise ValueError("A table schema must be provided")
            self._table = self._connection.create_table(
                name=table_name, schema=table_schema
            )
        if create_index:
            self._table.create_index(index_type=indexing_strategy, **indexing_kwargs)

    @classmethod
    def from_documents(cls) -> "LanceMultiModalStore":
        """Method to initialize the DB from multimodal documents"""

    @classmethod
    def from_data(cls) -> "LanceMultiModalStore":
        """Method to initialize the DB from LanceDB-compatible data types (Pandas, Polars, Arrow, Pydantic...)"""
