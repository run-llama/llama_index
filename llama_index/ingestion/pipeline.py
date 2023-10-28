from typing import Any, List, Optional, Sequence, cast

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.embeddings.utils import resolve_embed_model
from llama_index.indices.service_context import ServiceContext
from llama_index.ingestion.client import (
    ConfigurableDataSinkNames,
    ConfigurableDataSourceNames,
    ConfigurableTransformationNames,
    ConfiguredTransformationItem,
    DataSinkCreate,
    DataSourceCreate,
    Pipeline,
    PipelineCreate,
    Project,
    ProjectCreate,
)
from llama_index.ingestion.client.client import PlatformApi
from llama_index.ingestion.data_sinks import ConfigurableDataSinks, ConfiguredDataSink
from llama_index.ingestion.data_sources import (
    ConfigurableDataSources,
    ConfiguredDataSource,
)
from llama_index.ingestion.transformations import (
    ConfigurableTransformations,
    ConfiguredTransformation,
)
from llama_index.node_parser import SentenceAwareNodeParser
from llama_index.readers.base import ReaderConfig
from llama_index.schema import BaseComponent, BaseNode, Document, TransformComponent
from llama_index.vector_stores.types import BasePydanticVectorStore

DEFAULT_PIPELINE_NAME = "pipeline"
DEFAULT_PROJECT_NAME = "project"
BASE_URL = "http://localhost:8000"


def deserialize_transformation_component(
    component_dict: dict, component_type: ConfigurableTransformationNames
) -> BaseComponent:
    component_cls = ConfigurableTransformations[component_type].value.component_type
    return component_cls.from_dict(component_dict)


def deserialize_source_component(
    component_dict: dict, component_type: ConfigurableDataSourceNames
) -> BaseComponent:
    component_cls = ConfigurableDataSources[component_type].value.component_type
    return component_cls.from_dict(component_dict)


def deserialize_sink_component(
    component_dict: dict, component_type: ConfigurableDataSinkNames
) -> BaseComponent:
    component_cls = ConfigurableDataSinks[component_type].value.component_type
    return component_cls.from_dict(component_dict)


def run_transformations(
    nodes: List[BaseNode],
    transformations: Sequence[TransformComponent],
    in_place: bool = True,
    **kwargs: Any,
) -> List[BaseNode]:
    """Run a series of transformations on a set of nodes.

    Args:
        nodes: The nodes to transform.
        transformations: The transformations to apply to the nodes.

    Returns:
        The transformed nodes.
    """
    if not in_place:
        nodes = list(nodes)

    for transform in transformations:
        nodes = transform(nodes, **kwargs)

    return nodes


async def arun_transformations(
    nodes: List[BaseNode],
    transformations: Sequence[TransformComponent],
    in_place: bool = True,
    **kwargs: Any,
) -> List[BaseNode]:
    """Run a series of transformations on a set of nodes.

    Args:
        nodes: The nodes to transform.
        transformations: The transformations to apply to the nodes.

    Returns:
        The transformed nodes.
    """
    if not in_place:
        nodes = list(nodes)

    for transform in transformations:
        nodes = await transform.acall(nodes, **kwargs)

    return nodes


class IngestionPipeline(BaseModel):
    """An ingestion pipeline that can be applied to data."""

    name: str = Field(
        default=DEFAULT_PIPELINE_NAME,
        description="Unique name of the ingestion pipeline",
    )
    project_name: str = Field(
        default=DEFAULT_PROJECT_NAME, description="Unique name of the project"
    )
    base_url: str = Field(default=BASE_URL, description="Base URL for the platform")

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
        name: str = DEFAULT_PIPELINE_NAME,
        project_name: str = DEFAULT_PROJECT_NAME,
        transformations: Optional[List[TransformComponent]] = None,
        reader: Optional[ReaderConfig] = None,
        documents: Optional[Sequence[Document]] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        base_url: str = BASE_URL,
    ) -> None:
        if transformations is None:
            transformations = self._get_default_transformations()

        configured_transformations: List[ConfiguredTransformation] = []
        for transformation in transformations:
            configured_transformations.append(
                ConfiguredTransformation.from_component(transformation)
            )

        super().__init__(
            name=name,
            project_name=project_name,
            configured_transformations=configured_transformations,
            transformations=transformations,
            reader=reader,
            documents=documents,
            vector_store=vector_store,
            base_url=base_url,
        )

    @classmethod
    def from_service_context(
        cls,
        service_context: ServiceContext,
        name: str = DEFAULT_PIPELINE_NAME,
        project_name: str = DEFAULT_PROJECT_NAME,
        reader: Optional[ReaderConfig] = None,
        documents: Optional[Sequence[Document]] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
    ) -> "IngestionPipeline":
        transformations = [
            *service_context.transformations,
            service_context.embed_model,
        ]

        return cls(
            name=name,
            project_name=project_name,
            transformations=transformations,
            reader=reader,
            documents=documents,
            vector_store=vector_store,
        )

    @classmethod
    def from_pipeline_name(
        cls,
        name: str,
        project_name: str = DEFAULT_PROJECT_NAME,
        base_url: str = BASE_URL,
    ) -> "IngestionPipeline":
        client = PlatformApi(base_url=base_url)

        projects: List[Project] = client.project.get_project_by_name_api_project_get(
            project_name=project_name
        )
        if len(projects) < 0:
            raise ValueError(f"Project with name {project_name} not found")

        project = projects[0]
        assert project.id is not None, "Project ID should not be None"

        pipelines: List[
            Pipeline
        ] = client.pipeline.get_pipeline_by_name_api_pipeline_get(
            pipeline_name=name, project_name=project_name
        )
        if len(pipelines) < 0:
            raise ValueError(f"Pipeline with name {name} not found")

        pipeline = pipelines[0]

        transformations: List[TransformComponent] = []
        for configured_transformation in pipeline.configured_transformations:
            component_dict = cast(dict, configured_transformation.component)
            transformation_component_type = (
                configured_transformation.configurable_transformation_type
            )
            transformation = deserialize_transformation_component(
                component_dict, transformation_component_type
            )
            transformations.append(transformation)

        documents = []
        readers = []
        for data_source in pipeline.data_sources:
            component_dict = cast(dict, data_source.component)
            source_component_type = data_source.source_type
            source_component = deserialize_source_component(
                component_dict, source_component_type
            )

            if data_source.source_type == ConfigurableDataSourceNames.READER:
                readers.append(source_component)
            else:
                documents.append(source_component)

        vector_stores = []
        for data_sink in pipeline.data_sinks:
            if data_sink.sink_type in ConfigurableDataSinkNames:
                component_dict = cast(dict, data_sink.component)
                sink_component_type = data_sink.sink_type
                sink_component = deserialize_sink_component(
                    component_dict, sink_component_type
                )
                vector_stores.append(sink_component)

        return cls(
            name=name,
            project_name=project_name,
            transformations=transformations,
            reader=readers[0] if len(readers) > 0 else None,
            documents=documents,
            vector_store=vector_stores[0] if len(vector_stores) > 0 else None,
            base_url=base_url,
        )

    def _get_default_transformations(self) -> List[TransformComponent]:
        return [
            SentenceAwareNodeParser(),
            resolve_embed_model("default"),
        ]

    def register(
        self,
        verbose: bool = True,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[BaseNode]] = None,
    ) -> str:
        client = PlatformApi(base_url=BASE_URL)

        input_nodes: List[BaseNode] = cast(List[BaseNode], self.documents) or []
        if documents is not None:
            input_nodes += cast(List[BaseNode], documents)
        if nodes is not None:
            input_nodes += nodes

        configured_transformations: List[ConfiguredTransformationItem] = []
        for item in self.configured_transformations:
            name = ConfigurableTransformationNames[
                item.configurable_transformation_type.name
            ]
            configured_transformations.append(
                ConfiguredTransformationItem(
                    transformation_name=name,
                    component=item.component,
                    configurable_transformation_type=item.configurable_transformation_type.name,
                )
            )

            # remove callback manager
            configured_transformations[-1].component.pop("callback_manager", None)  # type: ignore

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
                input_nodes += documents

        for node in input_nodes:
            configured_data_source = ConfiguredDataSource.from_component(node)
            source_type = ConfigurableDataSourceNames[
                configured_data_source.configurable_data_source_type.name
            ]
            data_sources.append(
                DataSourceCreate(
                    name=configured_data_source.name,
                    source_type=source_type,
                    component=node,
                )
            )

        project = client.project.upsert_project_api_project_put(
            request=ProjectCreate(name=self.project_name)
        )
        assert project.id is not None, "Project ID should not be None"

        # upload
        pipeline = client.project.upsert_pipeline_for_project(
            project.id,
            request=PipelineCreate(
                name=self.name,
                configured_transformations=configured_transformations,
                data_sinks=data_sinks,
                data_sources=data_sources,
            ),
        )
        assert pipeline.id is not None, "Pipeline ID should not be None"

        # Print playground URL if not running remote
        if verbose:
            print(
                f"Pipeline available at: http://localhost:3000/playground/{pipeline.id}"
            )

        return pipeline.id

    def run_remote(
        self,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[BaseNode]] = None,
    ) -> str:
        client = PlatformApi(base_url=BASE_URL)

        input_nodes: List[BaseNode] = []
        if documents is not None:
            input_nodes += documents

        if nodes is not None:
            input_nodes += nodes

        if self.documents is not None:
            input_nodes += self.documents

        pipeline_id = self.register(verbose=False)

        # start pipeline?
        # the `PipeLineExecution` object should likely generate a URL at some point
        pipeline_execution = client.pipeline.create_configured_transformation_execution(
            pipeline_id
        )

        assert (
            pipeline_execution.id is not None
        ), "Pipeline execution ID should not be None"

        print(
            "Find your remote results here: https://llamalink.llamaindex.ai/"
            f"pipelines/execution?id={pipeline_execution.id}"
        )

        return pipeline_execution.id

    def run_local(
        self,
        show_progress: bool = False,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[BaseNode]] = None,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        input_nodes: List[BaseNode] = []
        if documents is not None:
            input_nodes += documents

        if nodes is not None:
            input_nodes += nodes

        if self.documents is not None:
            input_nodes += self.documents

        if self.reader is not None:
            input_nodes += self.reader.read()

        nodes = run_transformations(
            input_nodes,
            self.transformations,
            show_progress=show_progress,
            **kwargs,
        )

        if self.vector_store is not None:
            self.vector_store.add([n for n in nodes if n.embedding is not None])

        return nodes
