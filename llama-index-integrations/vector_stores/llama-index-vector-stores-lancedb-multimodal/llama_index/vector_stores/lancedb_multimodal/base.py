import lancedb
import os
import polars as pl
import pandas as pd
import io
import pyarrow as pa
import warnings
import httpx

from PIL import Image
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field
from lancedb import DBConnection
from lancedb.table import Table
from lancedb.rerankers import Reranker

from typing import Optional, Dict, Any, Sequence, Union, Literal, List
from llama_index.core.llms import ImageBlock
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQueryResult,
)
from llama_index.core.schema import Document, ImageDocument
from llama_index.core.bridge.pydantic import PrivateAttr
from .utils import (
    LanceDBMultiModalModel,
    LanceDBTextModel,
    get_lancedb_multimodal_embedding_model,
    get_lancedb_text_embedding_model,
)

DEFAULT_TABLE_NAME = "default_table"
DEFAULT_VECTOR_FIELD_NAME = "vector"
DEFAULT_TEXT_FIELD_NAME = "text"
DEFAULT_IMAGE_FIELD_NAME = "image"
DEFAULT_IMAGE_URL_FIELD_NAME = "image_url"


class LanceMultiModalStore(BasePydanticVectorStore):
    """
    Implementation of the MultiModal AI LakeHouse by LanceDB.
    """

    class Config:
        arbitrary_types_allowed = True

    _embedding_model: Optional[Union[LanceDBMultiModalModel, LanceDBTextModel]] = (
        PrivateAttr(default=None)
    )
    _table_schema: Union[LanceModel, pa.Schema] = PrivateAttr()
    _connection: DBConnection = PrivateAttr()
    _table: Table = PrivateAttr()
    _reranker: Optional[Reranker] = PrivateAttr(default=None)

    def __init__(
        self,
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
    ) -> None:
        self._reranker = reranker
        if connection:
            self._connection = connection
        elif uri and uri.startswith("db://") and not api_key:
            api_key = os.getenv("LANCEDB_API_KEY", None)
            if not api_key:
                raise ValueError(
                    "Non-local URI passed but API key neither provided nor set as an environmental variable."
                )
            if not region:
                region = "us-east-1"
            self._connection = lancedb.connect(uri=uri, region=region, api_key=api_key)
        elif uri and uri.startswith("db://") and api_key:
            if not region:
                region = "us-east-1"
            self._connection = lancedb.connect(uri=uri, region=region, api_key=api_key)
        elif uri and not uri.startswith("db://"):
            self._connection = lancedb.connect(uri=uri)
        else:
            raise ValueError(
                "A DBConnection object has not been provided and the you did not provide the URI for a local connection."
            )
        if text_embedding_model:
            self._embedding_model = get_lancedb_text_embedding_model(
                embedding_model=text_embedding_model, **embedding_model_kwargs
            )

            class TextSchema(LanceModel):
                id: str
                text: str = self._embedding_model.embedding_model.SourceField()
                vector: Vector(self._embedding_model.embedding_model.ndims()) = (
                    self._embedding_model.embedding_model.VectorField()
                )

            self._table_schema = TextSchema
        elif multimodal_embedding_model:
            self._embedding_model = get_lancedb_multimodal_embedding_model(
                embedding_model=multimodal_embedding_model, **embedding_model_kwargs
            )

            class MultiModalSchema(LanceModel):
                id: str
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
        else:
            raise ValueError(
                "One of multimodal_embedding_model and text_embedding_model has to be set."
            )
        self._table = self._connection.create_table(
            table_name, schema=self._table_schema
        )
        if indexing != "NO_INDEXING":
            self._table.create_index(index_type=indexing, **indexing_kwargs)

    @classmethod
    def from_table(
        cls,
        table_name: str,
        connection: Optional[DBConnection] = None,
        uri: Optional[str] = None,
        region: Optional[str] = None,
        api_key: Optional[str] = None,
        reranker: Optional[Reranker] = None,
    ) -> "LanceMultiModalStore":
        """
        Generate a LanceMultiModalStore from an existing table.
        """
        cls._reranker = reranker
        if connection:
            cls._connection = connection
        elif uri and uri.startswith("db://") and not api_key:
            api_key = os.getenv("LANCEDB_API_KEY", None)
            if not api_key:
                raise ValueError(
                    "Non-local URI passed but API key neither provided nor set as an environmental variable."
                )
            if not region:
                region = "us-east-1"
            cls._connection = lancedb.connect(uri=uri, region=region, api_key=api_key)
        elif uri and uri.startswith("db://") and api_key:
            if not region:
                region = "us-east-1"
            cls._connection = lancedb.connect(uri=uri, region=region, api_key=api_key)
        elif uri and not uri.startswith("db://"):
            cls._connection = lancedb.connect(uri=uri)
        else:
            raise ValueError(
                "A DBConnection object has not been provided and the you did not provide the URI for a local connection."
            )
        cls._table = cls._connection.open_table(table_name)
        cls._embedding_model = None
        cls._table_schema = cls._table.schema

        if "id" not in cls._table_schema.field_names():
            raise ValueError("The table must contain an 'id' field in their schema.")

        return cls

    @classmethod
    def from_documents(
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
    ) -> "LanceMultiModalStore":
        """
        Generate a LanceMultiModalStore from LlamaIndex Documents.
        """
        try:
            vector_store = cls(
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
            )
        except ValueError as e:
            raise ValueError(
                f"Initialization of the vector store from documents are failed: {e}"
            )
        data: List[dict] = []
        if text_embedding_model:
            assert all(isinstance(document, Document) for document in documents)
            for document in documents:
                if document.text:
                    data.append({"id": document.id_, "text": document.text})
                else:
                    warnings.warn(
                        f"Document {document.doc_id} does not contain text and has thus been skipped",
                        UserWarning,
                    )
        else:
            assert all(isinstance(document, ImageDocument) for document in documents)
            for document in documents:
                label = document.metadata.get("image_label", None) or ""
                if document.image:
                    data.append(
                        {
                            "id": document.id_,
                            "image_bytes": document.image,
                            "image_uri": document.image_url or "",
                            "label": label,
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
                        }
                    )
                else:
                    warnings.warn(
                        f"Document {document.doc_id} does not contain an image and has thus been skipped",
                        UserWarning,
                    )
        vector_store._table.add(data)
        return vector_store

    @classmethod
    def from_data(
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
    ) -> "LanceMultiModalStore":
        """
        Generate a LanceMultiModalStore from Pandas, Polars or PyArrow data.
        """
        try:
            vector_store = cls(
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
            )
        except ValueError as e:
            raise ValueError(
                f"Initialization of the vector store from documents are failed: {e}"
            )
        vector_store._table.add(data)
        return vector_store

    def query(self, query: Any, **kwargs: Any):
        raise NotImplementedError(
            "This method has not been implemented, please use 'query_text' or 'query_multimodal' instead."
        )

    def query_text(self, query: str, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query a text-based table.

        Args:
            query (str): a string representing a text query.

        """
        data = self._table.search(query).to_list()
        documents: List[Union[Document, ImageDocument]] = []
        for d in data:
            if "text" in d:
                documents.append(Document(text=d["text"], id_=d["id"]))
            else:
                documents.append(
                    ImageDocument(
                        image_url=d["image_uri"],
                        image=d["image_bytes"],
                        id_=d["id"],
                        metadata={"label": d["label"]},
                    )
                )
        return VectorStoreQueryResult(
            nodes=documents, ids=[doc.id_ for doc in documents]
        )

    def query_multimodal(
        self, query: Union[ImageDocument, Image.Image, str, ImageBlock], **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Query a multimodal table.

        Args:
            query (Union[ImageDocument, Image.Image, str, ImageBlock]): An ImageDocument or an ImageBlock, a PIL Image or a string representing an image URL.

        """
        if isinstance(query, (ImageBlock, ImageDocument)):
            image_buffer = query.resolve_image()
            image_query = Image.open(image_buffer)
        elif isinstance(query, Image.Image):
            image_query = query
        elif isinstance(query, str):
            image_bytes = httpx.get(query).content
            image_query = Image.open(io.BytesIO(image_bytes))
        else:
            raise ValueError("Image type not supported.")
        data = self._table.search(image_query).to_list()
        documents: List[ImageDocument] = []
        for d in data:
            documents.append(
                ImageDocument(
                    image_url=d["image_uri"],
                    image=d["image_bytes"],
                    id_=d["id"],
                    metadata={"label": d["label"]},
                )
            )
        return VectorStoreQueryResult(
            nodes=documents, ids=[doc.id_ for doc in documents]
        )

    def add(self, nodes: Any, **kwargs: Any):
        """
        This method has not been implemented, please use 'add_documents' or 'add_data' instead.
        """
        raise NotImplementedError(
            "This method has not been implemented, please use 'add_documents' or 'add_data' instead."
        )

    def add_documents(
        self, documents: Sequence[Union[Document, ImageDocument]], **kwargs: Any
    ) -> None:
        data: List[dict] = []
        if isinstance(self._embedding_model, LanceDBTextModel):
            assert all(isinstance(document, Document) for document in documents)
            for document in documents:
                if document.text:
                    data.append({"id": document.id_, "text": document.text})
                else:
                    warnings.warn(
                        f"Document {document.doc_id} does not contain text and has thus been skipped",
                        UserWarning,
                    )
        else:
            assert all(isinstance(document, ImageDocument) for document in documents)
            for document in documents:
                label = document.metadata.get("image_label", None) or ""
                if document.image:
                    data.append(
                        {
                            "id": document.id_,
                            "image_bytes": document.image,
                            "image_uri": document.image_url or "",
                            "label": label,
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
                        }
                    )
                else:
                    warnings.warn(
                        f"Document {document.doc_id} does not contain an image and has thus been skipped",
                        UserWarning,
                    )

        self._table.add(data)

    def add_data(
        self, data: Union[List[dict], pl.DataFrame, pd.DataFrame, pa.Table]
    ) -> None:
        self._table.add(data)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._table.delete(where="id = '" + ref_doc_id + "'")

    def delete_data(self, ref_doc_ids: List[str]) -> None:
        delete_where = "id IN ('" + "', '".join(ref_doc_ids) + "')"
        self._table.delete(where=delete_where)

    def delete_nodes(self, node_ids: Any, filters: Any, **delete_kwargs: Any):
        """
        This method has not been implemented, please use 'delete_data' instead.
        """
        raise NotImplementedError(
            "This method has not been implemented, please use 'delete_data' instead."
        )

    def clear(self) -> None:
        # restoring to the first version, i.e. the version where an empty table has been created
        self._table.restore(1)

    def get_data(self, ids: List[str]) -> List[Union[Document, ImageDocument]]:
        select_where = "id IN ('" + "', '".join(ids) + "')"
        data = self._table.search().where(select_where).select(columns=None).to_list()
        documents: List[Union[Document, ImageDocument]] = []
        for d in data:
            if "text" in d:
                documents.append(Document(text=d["text"], id_=d["id"]))
            else:
                documents.append(
                    ImageDocument(
                        image_url=d["image_uri"],
                        image=d["image_bytes"],
                        id_=d["id"],
                        metadata={"label": d["label"]},
                    )
                )
        return documents

    def get_nodes(self, node_ids: Any, filters: Any):
        """
        This method has not been implemented, please use 'get_data' instead.
        """
        raise NotImplementedError(
            "This method has not been implemented, please use 'get_data' instead."
        )
