from typing import Callable, List, Optional, Sequence

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.embeddings.base import BaseEmbedding
from llama_index.embeddings.utils import EmbedType, resolve_embed_model
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
from llama_index.llms.base import LLM
from llama_index.llms.utils import LLMType, resolve_llm
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    MetadataFeatureExtractor,
)
from llama_index.readers.base import ReaderConfig
from llama_index.schema import BaseComponent, BaseNode, Document, MetadataMode
from llama_index.vector_stores.types import BasePydanticVectorStore, NodeWithEmbedding


class IngestionPipeline(BaseModel):
    """An ingestion pipeline that can be applied to data."""

    name: str = Field(description="Unique name of the ingestion pipeline")
    configured_transformations: List[ConfiguredTransformation] = Field(
        description="Serialized schemas of transformations to apply to the data"
    )

    llm: LLM = Field(description="LLM to use to process the data")
    embed_model: BaseEmbedding = Field(
        description="Embedding model to use to embed the data"
    )

    transformations: List[BaseComponent] = Field(
        description="Transformations to apply to the data"
    )

    documents: Optional[Sequence[Document]] = Field(description="Documents to ingest")
    reader: Optional[ReaderConfig] = Field(description="Reader to use to read the data")
    vector_store: Optional[BasePydanticVectorStore] = Field(
        description="Vector store to use to store the data"
    )

    def __init__(
        self,
        name: Optional[str] = "llamaindex_pipeline",
        transformations: Optional[List[BaseComponent]] = None,
        reader: Optional[ReaderConfig] = None,
        documents: Optional[Sequence[Document]] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        llm: Optional[LLMType] = "default",
        embed_model: Optional[EmbedType] = "default",
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
            llm=resolve_llm(llm),
            embed_model=resolve_embed_model(embed_model),
        )

    def _get_default_transformations(self) -> List[BaseComponent]:
        return [
            SimpleNodeParser.from_defaults(),
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
        self, run_embeddings: bool = True, show_progress: bool = False
    ) -> Sequence[BaseNode]:
        inputs: List[Document] = []
        if self.documents is not None:
            inputs += self.documents

        if self.reader is not None:
            inputs += self.reader.read()

        pipeline = self._build_pipeline(show_progress=show_progress)

        nodes = pipeline(inputs)

        if run_embeddings:
            texts = [
                node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes
            ]
            embeddings = self.embed_model.get_text_embedding_batch(
                texts, show_progress=show_progress
            )

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding

        if self.vector_store is not None:
            self.vector_store.add(
                [
                    NodeWithEmbedding(node=n, embedding=n.embedding)
                    for n in nodes
                    if n.embedding is not None
                ]
            )

        return nodes

    def _build_pipeline(
        self, show_progress: bool = False
    ) -> Callable[[Sequence[Document]], Sequence[BaseNode]]:
        metadata_extractor = None
        extractors: List[MetadataFeatureExtractor] = []
        node_parser = SimpleNodeParser.from_defaults()

        for transformation in self.transformations:
            if isinstance(transformation, NodeParser):
                node_parser = transformation
            elif isinstance(transformation, MetadataExtractor):
                metadata_extractor = transformation
            elif isinstance(transformation, MetadataFeatureExtractor):
                extractors.append(transformation)
                extractors[-1].show_progress = show_progress

        if metadata_extractor is None:
            metadata_extractor = MetadataExtractor(extractors=extractors)

        node_parser.metadata_extractor = metadata_extractor

        # right now, local ingestion pipelines are just node parsers
        def pipeline_fn(documents: Sequence[Document]) -> Sequence[BaseNode]:
            return node_parser.get_nodes_from_documents(
                documents, show_progress=show_progress
            )

        return pipeline_fn
