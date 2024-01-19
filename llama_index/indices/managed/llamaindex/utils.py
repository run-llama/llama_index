import os
from typing import List, Optional

from llama_index_client import (
    ConfigurableDataSourceNames,
    ConfigurableTransformationNames,
    ConfiguredTransformationItem,
    DataSourceCreate,
    PipelineCreate,
    PipelineType,
    ProjectCreate,
)
from llama_index_client.client import AsyncPlatformApi, PlatformApi

from llama_index.embeddings import OpenAIEmbedding
from llama_index.ingestion.data_sources import (
    ConfiguredDataSource,
)
from llama_index.ingestion.pipeline import (
    DEFAULT_APP_URL,
    DEFAULT_BASE_URL,
    DEFAULT_PROJECT_NAME,
)
from llama_index.ingestion.transformations import (
    ConfiguredTransformation,
)
from llama_index.node_parser import SentenceSplitter
from llama_index.readers.base import ReaderConfig
from llama_index.schema import BaseNode, TransformComponent


def default_transformations() -> List[TransformComponent]:
    """Default transformations."""
    return [
        SentenceSplitter(),
        OpenAIEmbedding(),
    ]


def get_client(
    platform_api_key: Optional[str] = None,
    platform_base_url: Optional[str] = None,
    platform_app_url: Optional[str] = None,
    timeout: int = 60,
) -> PlatformApi:
    """Get the sync platform API client."""
    platform_base_url = platform_base_url or os.environ.get(
        "PLATFORM_BASE_URL", DEFAULT_BASE_URL
    )
    platform_app_url = platform_app_url or os.environ.get(
        "PLATFORM_APP_URL", DEFAULT_APP_URL
    )
    platform_api_key = platform_api_key or os.environ.get("PLATFORM_API_KEY", None)

    return PlatformApi(
        base_url=platform_base_url, token=platform_api_key, timeout=timeout
    )


def get_aclient(
    platform_api_key: Optional[str] = None,
    platform_base_url: Optional[str] = None,
    platform_app_url: Optional[str] = None,
    timeout: int = 60,
) -> AsyncPlatformApi:
    """Get the async platform API client."""
    platform_base_url = platform_base_url or os.environ.get(
        "PLATFORM_BASE_URL", DEFAULT_BASE_URL
    )
    platform_app_url = platform_app_url or os.environ.get(
        "PLATFORM_APP_URL", DEFAULT_APP_URL
    )
    platform_api_key = platform_api_key or os.environ.get("PLATFORM_API_KEY", None)

    return AsyncPlatformApi(
        base_url=platform_base_url, token=platform_api_key, timeout=timeout
    )


def get_pipeline_create(
    pipeline_name: str,
    client: PlatformApi,
    pipeline_type: PipelineType,
    project_name: str = DEFAULT_PROJECT_NAME,
    transformations: Optional[List[TransformComponent]] = None,
    readers: Optional[List[ReaderConfig]] = None,
    input_nodes: Optional[List[BaseNode]] = None,
) -> PipelineCreate:
    """Get a pipeline create object."""
    transformations = transformations or []
    readers = readers or []
    input_nodes = input_nodes or []

    configured_transformations: List[ConfiguredTransformation] = []
    for transformation in transformations:
        try:
            configured_transformations.append(
                ConfiguredTransformation.from_component(transformation)
            )
        except ValueError:
            raise ValueError(f"Unsupported transformation: {type(transformation)}")

    configured_transformation_items: List[ConfiguredTransformationItem] = []
    for item in configured_transformations:
        name = ConfigurableTransformationNames[
            item.configurable_transformation_type.name
        ]
        configured_transformation_items.append(
            ConfiguredTransformationItem(
                transformation_name=name,
                component=item.component,
                configurable_transformation_type=item.configurable_transformation_type.name,
            )
        )

        # remove callback manager
        configured_transformation_items[-1].component.pop("callback_manager", None)  # type: ignore

    data_sources = []
    for reader in readers:
        if reader.reader.is_remote:
            configured_data_source = ConfiguredDataSource.from_component(
                reader,
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
            documents = reader.read()
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

    project = client.project.upsert_project(request=ProjectCreate(name=project_name))
    assert project.id is not None, "Project ID should not be None"

    # upload
    return PipelineCreate(
        name=pipeline_name,
        configured_transformations=configured_transformation_items,
        data_sources=data_sources,
        data_sinks=[],
        pipeline_type=pipeline_type,
    )
