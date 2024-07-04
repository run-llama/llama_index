from typing import List, Optional

from llama_cloud import (
    ConfigurableTransformationNames,
    ConfiguredTransformationItem,
    PipelineCreate,
    PipelineType,
    ProjectCreate,
)
from llama_cloud.client import LlamaCloud

from llama_index.core.constants import (
    DEFAULT_PROJECT_NAME,
)
from llama_index.core.ingestion.transformations import (
    ConfiguredTransformation,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import ReaderConfig
from llama_index.core.schema import BaseNode, TransformComponent


def default_transformations() -> List[TransformComponent]:
    """Default transformations."""
    from llama_index.embeddings.openai import OpenAIEmbedding  # pants: no-infer-dep

    return [
        SentenceSplitter(),
        OpenAIEmbedding(),
    ]


def get_pipeline_create(
    pipeline_name: str,
    client: LlamaCloud,
    pipeline_type: PipelineType,
    project_name: str = DEFAULT_PROJECT_NAME,
    transformations: Optional[List[TransformComponent]] = None,
    readers: Optional[List[ReaderConfig]] = None,
    input_nodes: Optional[List[BaseNode]] = None,
) -> PipelineCreate:
    """Get a pipeline create object."""
    transformations = transformations or []

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

    project = client.projects.upsert_project(request=ProjectCreate(name=project_name))
    assert project.id is not None, "Project ID should not be None"

    # upload
    return PipelineCreate(
        name=pipeline_name,
        configured_transformations=configured_transformation_items,
        pipeline_type=pipeline_type,
        # we are uploading document dicrectly, so we don't need llama parse
        llama_parse_enabled=False,
    )
