from typing import List, Optional, Sequence, Union

from llama_index.bridge.pydantic import BaseModel, Field

from llama_index.indices.base import BaseIndex
from llama_index.ingestion.transformation import (
    ConfiguredTransformation,
    get_configured_transform,
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.readers.base import BasePydanticReader, BaseReader
from llama_index.schema import BaseComponent, Document
from llama_index.vector_stores.types import BasePydanticVectorStore


class IngestionPipeline(BaseModel):
    """An ingestion pipeline that can be applied to data."""

    name: str = Field(description="Unique name of the ingestion pipeline")
    transformations: list[ConfiguredTransformation] = Field(
        description="Transformations to apply to the data"
    )

    documents: Optional[Sequence[Document]] = Field(description="Documents to ingest")
    reader: Optional[BasePydanticReader] = Field(
        description="Reader to use to read the data"
    )
    vector_store: Optional[BasePydanticVectorStore] = Field(
        description="Vector store to use to store the data"
    )
    index_cls: Optional[BaseIndex] = Field(
        description="Index class to use to index the data"
    )

    def __init__(
        self,
        name: Optional[str] = "llamaindex_pipeline",
        transformations: Optional[list[BaseComponent]] = None,
        reader: Optional[Union[BaseReader, BasePydanticReader]] = None,
        documents: Optional[Sequence[Document]] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        index_cls: Optional[BaseIndex] = None,
    ) -> None:
        if documents is None and reader is None:
            raise ValueError("Must provide either documents or a reader")

        if transformations is None:
            transformations = self._get_default_transformations()

        configured_transformations: List[ConfiguredTransformation] = []
        for transformation in transformations:
            configured_transformations.append(
                get_configured_transform(transformation.to_dict())
            )

        super().__init__(
            name=name,
            transformations=configured_transformations,
            reader=reader,
            documents=documents,
            vector_store=vector_store,
            index_cls=index_cls,
        )

    def _get_default_transformations(self) -> List[BaseComponent]:
        return [
            SimpleNodeParser.from_defaults(),
        ]

    def run_remote(self) -> str:
        return "Find your remote results here: https://llamaindex.com/"

    def run_local(self) -> BaseIndex:
        pass  # TODO: How to do this?
