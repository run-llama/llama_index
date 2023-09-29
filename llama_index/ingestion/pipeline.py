from typing import Any, List, Optional, Sequence

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.embeddings.utils import resolve_embed_model
from llama_index.ingestion.client import (
    ConfiguredTransformationItem,
    DataSinkCreate,
    DataSourceCreate,
    ConfigurableDataSinkNames,
    ConfigurableDataSourceNames,
    ConfigurableTransformationNames,
)
from llama_index.ingestion.client.client import PlatformApi
from llama_index.ingestion.data_sinks import ConfiguredDataSink
from llama_index.ingestion.data_sources import ConfiguredDataSource
from llama_index.ingestion.transformations import ConfiguredTransformation
from llama_index.indices.service_context import ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.readers.base import ReaderConfig
from llama_index.schema import TransformComponent, BaseNode, Document
from llama_index.vector_stores.types import BasePydanticVectorStore

DEFAULT_PIPELINE_NAME = "llamaindex_pipeline"


class IngestionPipeline(BaseModel):
    """An ingestion pipeline that can be applied to data."""

    name: str = Field(description="Unique name of the ingestion pipeline")
    configured_transformations: List[ConfiguredTransformation] = Field(
        description="Serialized schemas of transformations to apply to the data"
    )

    transformations: List[TransformComponent] = Field(
        description="Transformations to apply to the data"
    )

    documents: Optional[Sequence[Document]] = Field(description="Documents to ingest")
    reader: Optional[ReaderConfig] = Field(description="Reader to use to read the data")
    vector_store: Optional[BasePydanticVectorStore] = Field(
        description="Vector store to use to store the data"
    )

    def __init__(
        self,
        name: Optional[str] = DEFAULT_PIPELINE_NAME,
        transformations: Optional[List[TransformComponent]] = None,
        reader: Optional[ReaderConfig] = None,
        documents: Optional[Sequence[Document]] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
    ) -> None:
        if documents is None and reader is None:
            raise ValueError("Must provide either documents or a reader")

        if transformations is None:
            transformations = self._get_default_transformations()

        configured_transformations: List[ConfiguredTransformation] = []
        for transformation in transformations:
            configured_transformations.append(
                ConfiguredTransformation.from_component(transformation)
            )

        super().__init__(
            name=name,
            configured_transformations=configured_transformations,
            transformations=transformations,
            reader=reader,
            documents=documents,
            vector_store=vector_store,
        )

    @classmethod
    def from_service_context(
        cls,
        service_context: ServiceContext,
        name: str = DEFAULT_PIPELINE_NAME,
        reader: Optional[ReaderConfig] = None,
        documents: Optional[Sequence[Document]] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
    ):
        transformations = [
            service_context.node_parser,
            service_context.embed_model,
        ]

        return cls(
            name=name,
            transformations=transformations,
            llm=service_context.llm_predictor.llm,
            reader=reader,
            documents=documents,
            vector_store=vector_store,
        )

    def _get_default_transformations(self) -> List[TransformComponent]:
        return [
            SimpleNodeParser.from_defaults(),
            resolve_embed_model("default"),
        ]

    def run_remote(
        self, pipeline_name: str = "pipeline", project_name: str = "llamaindex"
    ) -> str:
        client = PlatformApi(base_url="http://localhost:8000")

        configured_transformations: List[ConfiguredTransformationItem] = []
        for item in self.configured_transformations:
            name = ConfigurableTransformationNames[
                item.configurable_transformation_type.name
            ]
            configured_transformations.append(
                ConfiguredTransformationItem(
                    transformation_name=name, component=item.component
                )
            )

        data_sinks = []
        if self.vector_store is not None:
            configured_data_sink = ConfiguredDataSink.from_component(self.vector_store)
            sink_type = ConfigurableDataSinkNames[
                configured_data_sink.configurable_data_sink_type.name
            ]
            data_sinks.append(
                DataSinkCreate(
                    name=configured_data_sink.name,
                    sink_type=sink_type,
                    component=configured_data_sink.component,
                )
            )

        data_sources = []
        if self.reader is not None:
            if self.reader.reader.is_remote:
                configured_data_source = ConfiguredDataSource.from_component(
                    self.reader
                )
                source_type = ConfigurableDataSourceNames[
                    configured_data_source.configurable_data_source_type.name
                ]
                data_sources.append(
                    DataSourceCreate(
                        name=configured_data_source.name,
                        source_type=source_type,
                        component=configured_data_source.component,
                    )
                )
            else:
                documents = self.reader.read()
                if self.documents is not None:
                    documents += self.documents
                else:
                    self.documents = documents

        if self.documents is not None:
            for document in self.documents:
                configured_data_source = ConfiguredDataSource.from_component(document)
                source_type = ConfigurableDataSourceNames[
                    configured_data_source.configurable_data_source_type.name
                ]
                data_sources.append(
                    DataSourceCreate(
                        name=configured_data_source.name,
                        source_type=source_type,
                        component=document,
                    )
                )

        project = client.project.create_project_api_project_post(name=project_name)
        assert project.id is not None, "Project ID should not be None"

        # upload?
        pipeline = client.project.create_pipeline_for_project(
            name=pipeline_name,
            project_id=project.id,
            configured_transformations=configured_transformations,
            data_sinks=data_sinks,
            data_sources=data_sources,
        )
        assert pipeline.id is not None, "Pipeline ID should not be None"

        # start pipeline?
        # the `PipeLineExecution` object should likely generate a URL at some point
        pipeline_execution = client.pipeline.create_pipeline_execution(
            pipeline_id=pipeline.id
        )

        return f"Find your remote results here: {pipeline_execution.id}"

    def run_local(
        self, show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        nodes: List[BaseNode] = []
        if self.documents is not None:
            nodes += self.documents

        if self.reader is not None:
            nodes += self.reader.read()

        for transform in self.transformations:
            nodes = transform(nodes, show_progress=show_progress, **kwargs)

        if self.vector_store is not None:
            self.vector_store.add([n for n in nodes if n.embedding is not None])

        return nodes
