from lancedb.embeddings import get_registry, EmbeddingFunction
from lancedb.table import Table, AsyncTable
from lancedb.index import IvfPq, IvfFlat, HnswPq, HnswSq, BTree, Bitmap, LabelList, FTS
import httpx
from PIL import Image
import io
import os

from llama_index.core.bridge.pydantic import BaseModel, Field, model_validator
from llama_index.core.llms import ImageBlock
from llama_index.core.schema import ImageDocument, Document, NodeWithScore
from typing import Literal, Union, Any, Optional, Dict, List
from typing_extensions import Self


class CloudConnectionConfig(BaseModel):
    uri: str
    api_key: Optional[str]
    region: Optional[str]
    use_async: bool

    @model_validator(mode="after")
    def validate_connection(self) -> Self:
        if not self.api_key:
            self.api_key = os.getenv("LANCEDB_API_KEY", None)
            if not self.api_key:
                raise ValueError(
                    "You provided a cloud instance without setting the API key either in the code or as an environment variable."
                )
        if not self.region:
            self.region = "us-east-1"
        return self


class LocalConnectionConfig(BaseModel):
    uri: str
    use_async: bool


class TableConfig(BaseModel):
    table_name: str
    table_exists: bool


class EmbeddingConfig(BaseModel):
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
    ]
    multi_modal_embedding_model: Optional[
        Literal["open-clip", "colpali", "jina", "imagebind"]
    ]
    embedding_kwargs: Dict[str, Any]

    @model_validator(mode="after")
    def validate_embeddings(self) -> Self:
        if (
            self.text_embedding_model is None
            and self.multi_modal_embedding_model is None
        ):
            raise ValueError(
                "You must specify either a multimodal or a text embedding model"
            )
        if (
            self.text_embedding_model is not None
            and self.multi_modal_embedding_model is not None
        ):
            raise ValueError(
                "You cannot specify both a multimodal and a text embedding model"
            )


class IndexingConfig(BaseModel):
    indexing: Literal[
        "IVF_PQ",
        "IVF_HNSW_PQ",
        "IVF_HNSW_SQ",
        "FTS",
        "BTREE",
        "BITMAP",
        "LABEL_LIST",
        "NO_INDEXING",
    ]
    async_index_config: Optional[
        Union[IvfPq, IvfFlat, HnswPq, HnswSq, BTree, Bitmap, LabelList, FTS]
    ] = None
    indexing_kwargs: Dict[str, Any]

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        if self.indexing == "IVF_PQ":
            if not isinstance(self.async_index_config, IvfPq):
                self.async_index_config = IvfPq(**self.indexing_kwargs)
        elif self.indexing == "IVF_HNSW_PQ":
            if not isinstance(self.async_index_config, HnswPq):
                self.async_index_config = HnswPq(**self.indexing_kwargs)
        elif self.indexing == "IVF_HNSW_SQ":
            if not isinstance(self.async_index_config, HnswSq):
                self.async_index_config = HnswSq(**self.indexing_kwargs)
        elif self.indexing == "FTS":
            if not isinstance(self.async_index_config, FTS):
                self.async_index_config = FTS(**self.indexing_kwargs)
        elif self.indexing == "BTREE":
            if not isinstance(self.async_index_config, BTree):
                self.async_index_config = BTree(**self.indexing_kwargs)
        elif self.indexing == "BITMAP":
            if not isinstance(self.async_index_config, Bitmap):
                self.async_index_config = Bitmap(**self.indexing_kwargs)
        elif self.indexing == "LABEL_LIST":
            if not isinstance(self.async_index_config, LabelList):
                self.async_index_config = LabelList(**self.indexing_kwargs)
        elif self.indexing == "NO_INDEXING":
            self.async_index_config = None
        return self


class LanceDBTextModel(BaseModel):
    embedding_model: Union[
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
        ],
        EmbeddingFunction,
    ]
    kwargs: dict = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_embedding_model(self) -> Self:
        if isinstance(self.embedding_model, str):
            try:
                self.embedding_model = (
                    get_registry().get(self.embedding_model).create(**self.kwargs)
                )
            except Exception as e:
                raise ValueError(
                    f"An exception occurred while creating the embeddings function: {e}"
                )
        return self


class LanceDBMultiModalModel(BaseModel):
    embedding_model: Union[
        Literal["open-clip", "colpali", "jina", "imagebind"], EmbeddingFunction
    ]
    kwargs: dict = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_embedding_model(self) -> Self:
        if isinstance(self.embedding_model, str):
            try:
                self.embedding_model = (
                    get_registry().get(self.embedding_model).create(**self.kwargs)
                )
            except Exception as e:
                raise ValueError(
                    f"An exception occurred while creating the embeddings function: {e}"
                )
        return self


def get_lancedb_text_embedding_model(
    embedding_model: Literal[
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
    ],
    **kwargs: Any,
):
    """
    Get a pre-defined LanceDB text embedding model.

    Args:
        embedding_model (str): name of the embedding model.
        **kwargs (Any): keyword arguments that are necessary for the initialization of the embedding model you want to use.

    Returns:
        EmbeddingFunction: a LanceDB embedding function.

    """
    return LanceDBTextModel(embedding_model=embedding_model, kwargs=kwargs)


def get_lancedb_multimodal_embedding_model(
    embedding_model: Literal["open-clip", "colpali", "jina", "imagebind"], **kwargs: Any
):
    """
    Get a pre-defined LanceDB multimodal embedding model.

    Args:
        embedding_model (str): name of the embedding model.
        **kwargs (Any): keyword arguments that are necessary for the initialization of the embedding model you want to use.

    Returns:
        EmbeddingFunction: a LanceDB embedding function.

    """
    return LanceDBMultiModalModel(embedding_model=embedding_model, kwargs=kwargs)


def query_text(table: Table, query: str, **kwargs: Any) -> List[NodeWithScore]:
    """
    Query a text-based table.

    Args:
        query (str): a string representing a text query.

    """
    data = table.search(query).to_list()
    documents: List[NodeWithScore] = []
    for d in data:
        if "text" in d:
            documents.append(
                NodeWithScore(
                    node=Document(text=d["text"], id_=d["id"]), score=d["_distance"]
                )
            )
        else:
            documents.append(
                NodeWithScore(
                    ImageDocument(
                        image_url=d["image_uri"],
                        image=d["image_bytes"],
                        id_=d["id"],
                        metadata={"label": d["label"]},
                    ),
                    score=d["_distance"],
                )
            )
    return documents


def query_multimodal(
    table: Table,
    query: Union[ImageDocument, Image.Image, str, ImageBlock],
    **kwargs: Any,
) -> List[NodeWithScore]:
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
    data = table.search(image_query).to_list()
    documents: List[NodeWithScore] = []
    for d in data:
        documents.append(
            NodeWithScore(
                ImageDocument(
                    image_url=d["image_uri"],
                    image=d["image_bytes"],
                    id_=d["id"],
                    metadata={"label": d["label"]},
                ),
                score=d["_distance"],
            )
        )
    return documents


async def aquery_text(
    table: AsyncTable, query: str, **kwargs: Any
) -> List[NodeWithScore]:
    """
    Query a text-based table.

    Args:
        query (str): a string representing a text query.

    """
    dt = await table.search(query)
    data = await dt.to_list()
    documents: List[NodeWithScore] = []
    for d in data:
        if "text" in d:
            documents.append(
                NodeWithScore(
                    node=Document(text=d["text"], id_=d["id"]), score=d["_distance"]
                )
            )
        else:
            documents.append(
                NodeWithScore(
                    ImageDocument(
                        image_url=d["image_uri"],
                        image=d["image_bytes"],
                        id_=d["id"],
                        metadata={"label": d["label"]},
                    ),
                    score=d["_distance"],
                )
            )
    return documents


async def aquery_multimodal(
    table: AsyncTable,
    query: Union[ImageDocument, Image.Image, str, ImageBlock],
    **kwargs: Any,
) -> List[NodeWithScore]:
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
    dt = await table.search(image_query)
    data = await dt.to_list()
    documents: List[NodeWithScore] = []
    for d in data:
        documents.append(
            NodeWithScore(
                ImageDocument(
                    image_url=d["image_uri"],
                    image=d["image_bytes"],
                    id_=d["id"],
                    metadata={"label": d["label"]},
                ),
                score=d["_distance"],
            )
        )
    return documents
