import lancedb
import json
import polars as pl
import pandas as pd
import pyarrow as pa
import warnings
import httpx

from lancedb.pydantic import LanceModel, Vector
from pydantic import Field
from lancedb import DBConnection, AsyncConnection
from lancedb.table import Table, AsyncTable
from lancedb.rerankers import Reranker

from typing import Optional, Dict, Any, Sequence, Union, Literal, List, cast
from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.core.schema import Document, ImageDocument
from .utils import (
    LanceDBMultiModalModel,
    LanceDBTextModel,
    LocalConnectionConfig,
    CloudConnectionConfig,
    EmbeddingConfig,
    IndexingConfig,
    TableConfig,
    get_lancedb_multimodal_embedding_model,
    get_lancedb_text_embedding_model,
)
from .retriever import LanceDBRetriever
from .query_engine import LanceDBRetrieverQueryEngine

DEFAULT_TABLE_NAME = "default_table"


class LanceDBMultiModalIndex(BaseManagedIndex):
    """
    Implementation of the MultiModal AI LakeHouse by LanceDB.
    """

    class Config:
        arbitrary_types_allowed = True

    connection_config: Union[LocalConnectionConfig, CloudConnectionConfig]
    embedding_config: EmbeddingConfig
    indexing_config: IndexingConfig
    table_config: TableConfig

    _embedding_model: Optional[Union[LanceDBMultiModalModel, LanceDBTextModel]] = None
    _table_schema: Optional[Union[LanceModel, pa.Schema]] = None
    _connection: Optional[Union[DBConnection, AsyncConnection]] = None
    _table: Optional[Union[Table, AsyncTable]] = None
    _reranker: Optional[Reranker] = None

    def __init__(
        self,
        connection: Optional[Union[DBConnection, AsyncConnection]] = None,
        uri: Optional[str] = None,
        region: Optional[str] = None,
        api_key: Optional[str] = None,
        text_embedding_model: Optional[
            Literal[
                "bedrock-text",
                "cohere",
                "gemini-text",
                "instructor",
                "ollama",
                "openai",
                "sentence-transformers",
                "gte-text",
                "huggingface",
                "colbert",
                "jina",
                "watsonx",
                "voyageai",
            ]
        ] = None,
        multimodal_embedding_model: Optional[
            Literal["open-clip", "colpali", "jina", "imagebind"]
        ] = None,
        embedding_model_kwargs: Dict[str, Any] = {},
        table_name: str = DEFAULT_TABLE_NAME,
        indexing: Literal[
            "IVF_PQ",
            "IVF_HNSW_PQ",
            "IVF_HNSW_SQ",
            "FTS",
            "BTREE",
            "BITMAP",
            "LABEL_LIST",
            "NO_INDEXING",
        ] = "IVF_PQ",
        indexing_kwargs: Dict[str, Any] = {},
        reranker: Optional[Reranker] = None,
        use_async: bool = False,
        table_exists: bool = False,
    ) -> None:
        self._reranker = reranker
        if connection:
            assert isinstance(connection, (DBConnection, AsyncConnection)), (
                "You did not provide a valid LanceDB connection"
            )
            if use_async:
                assert isinstance(connection, AsyncConnection), (
                    "You set use_async to True, but you provided a synchronous connection"
                )
            else:
                assert isinstance(connection, DBConnection), (
                    "You set use_async to False, but you provided an asynchronous connection"
                )
            self._connection = connection
        elif uri and uri.startswith("db://"):
            self.connection_config = CloudConnectionConfig(
                uri=uri,
                api_key=api_key,
                region=region,
                use_async=use_async,
            )
        elif uri and not uri.startswith("db://"):
            self.connection_config = LocalConnectionConfig(
                uri=uri,
                use_async=use_async,
            )
        else:
            raise ValueError(
                "No connection has been passed and no URI has been set for local or remote connection"
            )
        self.embedding_config = EmbeddingConfig(
            text_embedding_model=text_embedding_model,
            multi_modal_embedding_model=multimodal_embedding_model,
            embedding_kwargs=embedding_model_kwargs,
        )
        self.indexing_config = IndexingConfig(
            indexing=indexing, indexing_kwargs=indexing_kwargs
        )
        self.table_config = TableConfig(
            table_name=table_name,
            table_exists=table_exists,
        )

    def create_index(self) -> None:
        if self._connection:
            return
        if self.connection_config.use_async:
            raise ValueError(
                "You are trying to establish a synchronous connection when use_async is set to True"
            )
        if isinstance(self.connection_config, LocalConnectionConfig):
            self._connection = lancedb.connect(uri=self.connection_config.uri)
        else:
            self._connection = lancedb.connect(
                uri=self.connection_config.uri,
                region=self.connection_config.region,
                api_key=self.connection_config.api_key,
            )

        self._connection = cast(DBConnection, self._connection)

        if self.embedding_config.text_embedding_model:
            self._embedding_model = get_lancedb_text_embedding_model(
                embedding_model=self.embedding_config.text_embedding_model,
                **self.embedding_config.embedding_kwargs,
            )

            class TextSchema(LanceModel):
                id: str
                metadata: str = Field(default=json.dumps({}))
                text: str = self._embedding_model.embedding_modxel.SourceField()
                vector: Vector(self._embedding_model.embedding_model.ndims()) = (
                    self._embedding_model.embedding_model.VectorField()
                )

            self._table_schema = TextSchema
        else:
            self._embedding_model = get_lancedb_multimodal_embedding_model(
                embedding_model=self.embedding_config.multi_modal_embedding_model,
                **self.embedding_config.embedding_kwargs,
            )

            class MultiModalSchema(LanceModel):
                id: str
                metadata: str = Field(default=json.dumps({}))
                label: str = Field(
                    default_factory=str,
                )
                image_uri: str = (
                    self._embedding_model.embedding_model.SourceField()
                )  # image uri as the source
                image_bytes: bytes = (
                    self._embedding_model.embedding_model.SourceField()
                )  # image bytes as the source
                vector: Vector(self._embedding_model.embedding_model.ndims()) = (
                    self._embedding_model.embedding_model.VectorField()
                )  # vector column
                vec_from_bytes: Vector(
                    self._embedding_model.embedding_model.ndims()
                ) = self._embedding_model.embedding_model.VectorField()  # Another vector column

            self._table_schema = MultiModalSchema

        if not self.table_config.table_exists:
            self._table = self._connection.create_table(
                self.table_config.table_name, schema=self._table_schema
            )
            if self.indexing_config.indexing != "NO_INDEXING":
                self._table.create_index(
                    index_type=self.indexing_config.indexing,
                    **self.indexing_config.indexing_kwargs,
                )
        else:
            self._table = self._connection.open_table(self.table_config.table_name)
            self._table_schema = self._table.schema

    async def acreate_index(self) -> None:
        if self._connection:
            return
        if not self.connection_config.use_async:
            raise ValueError(
                "You are trying to establish an asynchronous connection when use_async is set to False"
            )
        if isinstance(self.connection_config, LocalConnectionConfig):
            self._connection = await lancedb.connect_async(
                uri=self.connection_config.uri
            )
        else:
            self._connection = await lancedb.connect_async(
                uri=self.connection_config.uri,
                region=self.connection_config.region,
                api_key=self.connection_config.api_key,
            )
        self._connection = cast(AsyncConnection, self._connection)
        if self.embedding_config.text_embedding_model:
            self._embedding_model = get_lancedb_text_embedding_model(
                embedding_model=self.embedding_config.text_embedding_model,
                **self.embedding_config.embedding_kwargs,
            )

            class TextSchema(LanceModel):
                id: str
                metadata: str = Field(default=json.dumps({}))
                text: str = self._embedding_model.embedding_model.SourceField()
                vector: Vector(self._embedding_model.embedding_model.ndims()) = (
                    self._embedding_model.embedding_model.VectorField()
                )

            self._table_schema = TextSchema
        else:
            self._embedding_model = get_lancedb_multimodal_embedding_model(
                embedding_model=self.embedding_config.multi_modal_embedding_model,
                **self.embedding_config.embedding_kwargs,
            )
            self._embedding_model.validate_embedding_model()

            class MultiModalSchema(LanceModel):
                id: str
                metadata: str = Field(default=json.dumps({}))
                label: str = Field(
                    default_factory=str,
                )
                image_uri: str = (
                    self._embedding_model.embedding_model.SourceField()
                )  # image uri as the source
                image_bytes: bytes = (
                    self._embedding_model.embedding_model.SourceField()
                )  # image bytes as the source
                vector: Vector(self._embedding_model.embedding_model.ndims()) = (
                    self._embedding_model.embedding_model.VectorField()
                )  # vector column
                vec_from_bytes: Vector(
                    self._embedding_model.embedding_model.ndims()
                ) = self._embedding_model.embedding_model.VectorField()  # Another vector column

            self._table_schema = MultiModalSchema

        if not self.table_config.table_exists:
            self._table = await self._connection.create_table(
                self.table_config.table_name, schema=self._table_schema
            )
            if self.indexing_config.indexing != "NO_INDEXING":
                await self._table.create_index(
                    config=self.indexing_config.async_index_config,
                    column="vector",
                    **self.indexing_config.indexing_kwargs,
                )
        else:
            self._table = await self._connection.open_table(
                self.table_config.table_name
            )
            self._table_schema = await self._table.schema()

    @classmethod
    async def from_documents(
        cls,
        documents: Sequence[Union[Document, ImageDocument]],
        connection: Optional[DBConnection] = None,
        uri: Optional[str] = None,
        region: Optional[str] = None,
        api_key: Optional[str] = None,
        text_embedding_model: Optional[
            Literal[
                "bedrock-text",
                "cohere",
                "gemini-text",
                "instructor",
                "ollama",
                "openai",
                "sentence-transformers",
                "gte-text",
                "huggingface",
                "colbert",
                "jina",
                "watsonx",
                "voyageai",
            ]
        ] = None,
        multimodal_embedding_model: Optional[
            Literal["open-clip", "colpali", "jina", "imagebind"]
        ] = None,
        embedding_model_kwargs: Dict[str, Any] = {},
        table_name: str = DEFAULT_TABLE_NAME,
        indexing: Literal[
            "IVF_PQ",
            "IVF_HNSW_PQ",
            "IVF_HNSW_SQ",
            "FTS",
            "BTREE",
            "BITMAP",
            "LABEL_LIST",
            "NO_INDEXING",
        ] = "IVF_PQ",
        indexing_kwargs: Dict[str, Any] = {},
        reranker: Optional[Reranker] = None,
        use_async: bool = False,
        table_exists: bool = False,
    ) -> "LanceDBMultiModalIndex":
        """
        Generate a LanceDBMultiModalIndex from LlamaIndex Documents.
        """
        try:
            index = cls(
                connection,
                uri,
                region,
                api_key,
                text_embedding_model,
                multimodal_embedding_model,
                embedding_model_kwargs,
                table_name,
                indexing,
                indexing_kwargs,
                reranker,
                use_async,
                table_exists,
            )
        except ValueError as e:
            raise ValueError(
                f"Initialization of the index from documents are failed: {e}"
            )
        if use_async:
            await index.acreate_index()
        else:
            index.create_index()
        data: List[dict] = []
        if text_embedding_model:
            assert all(isinstance(document, Document) for document in documents)
            for document in documents:
                if document.text:
                    data.append(
                        {
                            "id": document.id_,
                            "text": document.text,
                            "metadata": json.dumps(document.metadata),
                        }
                    )
                else:
                    warnings.warn(
                        f"Document {document.doc_id} does not contain text and has thus been skipped",
                        UserWarning,
                    )
        else:
            assert all(isinstance(document, ImageDocument) for document in documents)
            for document in documents:
                label = json.dumps(document.metadata).get("image_label", None) or ""
                if document.image:
                    data.append(
                        {
                            "id": document.id_,
                            "image_bytes": document.image,
                            "image_uri": document.image_url or "",
                            "label": label,
                            "metadata": json.dumps(document.metadata),
                        }
                    )
                elif document.image_url:
                    image_bytes = httpx.get(document.image_url).content
                    data.append(
                        {
                            "id": document.id_,
                            "image_bytes": image_bytes,
                            "image_uri": document.image_url,
                            "label": label,
                            "metadata": json.dumps(document.metadata),
                        }
                    )
                elif document.image_path:
                    image_bytes = document.resolve_image().read()
                    data.append(
                        {
                            "id": document.id_,
                            "image_bytes": image_bytes,
                            "image_uri": document.image_url or "",
                            "label": label,
                            "metadata": json.dumps(document.metadata),
                        }
                    )
                else:
                    warnings.warn(
                        f"Document {document.doc_id} does not contain an image and has thus been skipped",
                        UserWarning,
                    )
        if use_async:
            await index._table.add(data)
        else:
            index._table.add(data)
        return index

    @classmethod
    async def from_data(
        cls,
        data: Union[List[dict], pa.Table, pl.DataFrame, pd.DataFrame],
        connection: Optional[DBConnection] = None,
        uri: Optional[str] = None,
        region: Optional[str] = None,
        api_key: Optional[str] = None,
        text_embedding_model: Optional[
            Literal[
                "bedrock-text",
                "cohere",
                "gemini-text",
                "instructor",
                "ollama",
                "openai",
                "sentence-transformers",
                "gte-text",
                "huggingface",
                "colbert",
                "jina",
                "watsonx",
                "voyageai",
            ]
        ] = None,
        multimodal_embedding_model: Optional[
            Literal["open-clip", "colpali", "jina", "imagebind"]
        ] = None,
        embedding_model_kwargs: Dict[str, Any] = {},
        table_name: str = DEFAULT_TABLE_NAME,
        indexing: Literal[
            "IVF_PQ",
            "IVF_HNSW_PQ",
            "IVF_HNSW_SQ",
            "FTS",
            "BTREE",
            "BITMAP",
            "LABEL_LIST",
            "NO_INDEXING",
        ] = "IVF_PQ",
        indexing_kwargs: Dict[str, Any] = {},
        reranker: Optional[Reranker] = None,
        use_async: bool = False,
        table_exists: bool = False,
    ) -> "LanceDBMultiModalIndex":
        """
        Generate a LanceDBMultiModalIndex from Pandas, Polars or PyArrow data.
        """
        try:
            index = cls(
                connection,
                uri,
                region,
                api_key,
                text_embedding_model,
                multimodal_embedding_model,
                embedding_model_kwargs,
                table_name,
                indexing,
                indexing_kwargs,
                reranker,
                use_async,
                table_exists,
            )
        except ValueError as e:
            raise ValueError(
                f"Initialization of the vector store from documents are failed: {e}"
            )
        if use_async:
            await index.acreate_index()
            await index._table.add(data)
        else:
            index.create_index()
            index._table.add(data)

        return index

    def as_retriever(self, **kwargs):
        if self.embedding_config.text_embedding_model:
            multimodal = False
        else:
            multimodal = True
        return LanceDBRetriever(
            table=self._table,
            multimodal=multimodal,
            **kwargs,
        )

    def as_query_engine(self, **kwargs):
        retriever = self.as_retriever()
        return LanceDBRetrieverQueryEngine(retriever=retriever, **kwargs)

    async def ainsert_nodes(
        self, documents: Sequence[Union[Document, ImageDocument]], **kwargs: Any
    ) -> None:
        data: List[dict] = []
        if isinstance(self._embedding_model, LanceDBTextModel):
            assert all(isinstance(document, Document) for document in documents)
            for document in documents:
                if document.text:
                    data.append(
                        {
                            "id": document.id_,
                            "text": document.text,
                            "metadata": json.dumps(document.metadata),
                        }
                    )
                else:
                    warnings.warn(
                        f"Document {document.doc_id} does not contain text and has thus been skipped",
                        UserWarning,
                    )
        else:
            assert all(isinstance(document, ImageDocument) for document in documents)
            for document in documents:
                label = json.dumps(document.metadata).get("image_label", None) or ""
                if document.image:
                    data.append(
                        {
                            "id": document.id_,
                            "image_bytes": document.image,
                            "image_uri": document.image_url or "",
                            "label": label,
                            "metadata": json.dumps(document.metadata),
                        }
                    )
                elif document.image_url:
                    image_bytes = httpx.get(document.image_url).content
                    data.append(
                        {
                            "id": document.id_,
                            "image_bytes": image_bytes,
                            "image_uri": document.image_url,
                            "label": label,
                            "metadata": json.dumps(document.metadata),
                        }
                    )
                elif document.image_path:
                    image_bytes = document.resolve_image().read()
                    data.append(
                        {
                            "id": document.id_,
                            "image_bytes": image_bytes,
                            "image_uri": document.image_url or "",
                            "label": label,
                            "metadata": json.dumps(document.metadata),
                        }
                    )
                else:
                    warnings.warn(
                        f"Document {document.doc_id} does not contain an image and has thus been skipped",
                        UserWarning,
                    )

        if self.connection_config.use_async:
            self._table = cast(AsyncTable, self._table)
            await self._table.add(data)
        else:
            raise ValueError(
                "Attempting to add documents asynchronously with a synchronous connection!"
            )

    def insert_nodes(
        self, documents: Sequence[Union[Document, ImageDocument]], **kwargs: Any
    ) -> None:
        data: List[dict] = []
        if isinstance(self._embedding_model, LanceDBTextModel):
            assert all(isinstance(document, Document) for document in documents)
            for document in documents:
                if document.text:
                    data.append(
                        {
                            "id": document.id_,
                            "text": document.text,
                            "metadata": json.dumps(document.metadata),
                        }
                    )
                else:
                    warnings.warn(
                        f"Document {document.doc_id} does not contain text and has thus been skipped",
                        UserWarning,
                    )
        else:
            assert all(isinstance(document, ImageDocument) for document in documents)
            for document in documents:
                label = json.dumps(document.metadata).get("image_label", None) or ""
                if document.image:
                    data.append(
                        {
                            "id": document.id_,
                            "image_bytes": document.image,
                            "image_uri": document.image_url or "",
                            "label": label,
                            "metadata": json.dumps(document.metadata),
                        }
                    )
                elif document.image_url:
                    image_bytes = httpx.get(document.image_url).content
                    data.append(
                        {
                            "id": document.id_,
                            "image_bytes": image_bytes,
                            "image_uri": document.image_url,
                            "label": label,
                            "metadata": json.dumps(document.metadata),
                        }
                    )
                elif document.image_path:
                    image_bytes = document.resolve_image().read()
                    data.append(
                        {
                            "id": document.id_,
                            "image_bytes": image_bytes,
                            "image_uri": document.image_url or "",
                            "label": label,
                            "metadata": json.dumps(document.metadata),
                        }
                    )
                else:
                    warnings.warn(
                        f"Document {document.doc_id} does not contain an image and has thus been skipped",
                        UserWarning,
                    )

        if not self.connection_config.use_async:
            self._table = cast(Table, self._table)
            self._table.add(data)
        else:
            raise ValueError(
                "Attempting to add documents synchronously with an asynchronous connection!"
            )

    def insert_data(
        self, data: Union[List[dict], pl.DataFrame, pd.DataFrame, pa.Table]
    ) -> None:
        if not self.connection_config.use_async:
            self._table = cast(Table, self._table)
            self._table.add(data)
        else:
            raise ValueError(
                "Attempting to add data asynchronously with a synchronous connection!"
            )

    async def ainsert_data(
        self, data: Union[List[dict], pl.DataFrame, pd.DataFrame, pa.Table]
    ) -> None:
        if self.connection_config.use_async:
            self._table = cast(AsyncTable, self._table)
            await self._table.add(data)
        else:
            raise ValueError(
                "Attempting to add data synchronously with an asynchronous connection!"
            )

    def insert(self, document: Union[Document, ImageDocument], **insert_kwargs):
        return self.insert_nodes(documents=[document], **insert_kwargs)

    async def ainsert(self, document: Union[Document, ImageDocument], **insert_kwargs):
        return await self.ainsert_nodes(documents=[document], **insert_kwargs)

    def delete_ref_doc(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        if not self.connection_config.use_async:
            self._table = cast(Table, self._table)
            self._table.delete(where="id = '" + ref_doc_id + "'")
        else:
            raise ValueError(
                "Attempting to delete data synchronously with an asynchronous connection!"
            )

    async def adelete_ref_doc(self, ref_doc_id: str, **delete_kwargs):
        if self.connection_config.use_async:
            self._table = cast(AsyncTable, self._table)
            await self._table.delete(where="id = '" + ref_doc_id + "'")
        else:
            raise ValueError(
                "Attempting to delete data asynchronously with a synchronous connection!"
            )

    def delete_nodes(self, ref_doc_ids: List[str]) -> None:
        if not self.connection_config.use_async:
            self._table = cast(Table, self._table)
            delete_where = "id IN ('" + "', '".join(ref_doc_ids) + "')"
            self._table.delete(where=delete_where)
        else:
            raise ValueError(
                "Attempting to delete data synchronously with an asynchronous connection!"
            )

    async def adelete_nodes(self, ref_doc_ids: List[str]) -> None:
        if self.connection_config.use_async:
            self._table = cast(AsyncTable, self._table)
            delete_where = "id IN ('" + "', '".join(ref_doc_ids) + "')"
            await self._table.delete(where=delete_where)
        else:
            raise ValueError(
                "Attempting to delete data asynchronously with a synchronous connection!"
            )

    def _insert(self, nodes: Any, **insert_kwargs: Any) -> Any:
        raise NotImplementedError("_insert is not implemented.")

    def update(self, document: Any, **update_kwargs: Any) -> Any:
        raise NotImplementedError("update is not implemented.")

    def update_ref_doc(self, document: Any, **update_kwargs: Any) -> Any:
        raise NotImplementedError("update_ref_doc is not implemented.")

    async def aupdate_ref_doc(self, document: Any, **update_kwargs: Any) -> Any:
        raise NotImplementedError("aupdate_ref_doc is not implemented.")

    def refresh(self, documents: Any, **update_kwargs: Any) -> Any:
        raise NotImplementedError("refresh is not implemented.")

    def refresh_ref_docs(self, documents: Any, **update_kwargs: Any) -> Any:
        raise NotImplementedError("refresh_ref_docs is not implemented.")

    async def arefresh_ref_docs(self, documents: Any, **update_kwargs: Any) -> Any:
        raise NotImplementedError("arefresh_ref_docs is not implemented.")
