from typing import Optional, Tuple

from llama_cloud import (
    AutoTransformConfig,
    Pipeline,
    PipelineCreateEmbeddingConfig,
    PipelineCreateEmbeddingConfig_OpenaiEmbedding,
    PipelineCreateTransformConfig,
    PipelineType,
    Project,
)
from llama_cloud.client import LlamaCloud


def default_embedding_config() -> PipelineCreateEmbeddingConfig:
    from llama_index.embeddings.openai import OpenAIEmbedding  # pants: no-infer-dep

    return PipelineCreateEmbeddingConfig_OpenaiEmbedding(
        type="OPENAI_EMBEDDING",
        component=OpenAIEmbedding(),
    )


def default_transform_config() -> PipelineCreateTransformConfig:
    return AutoTransformConfig()


def resolve_project(
    client: LlamaCloud,
    project_name: Optional[str],
    project_id: Optional[str],
    organization_id: Optional[str],
) -> Project:
    if project_id is not None:
        return client.projects.get_project(project_id=project_id)
    else:
        projects = client.projects.list_projects(
            project_name=project_name, organization_id=organization_id
        )
        if len(projects) == 0:
            raise ValueError(f"No project found with name {project_name}")
        elif len(projects) > 1:
            raise ValueError(
                f"Multiple projects found with name {project_name}. Please specify organization_id."
            )
        return projects[0]


def resolve_pipeline(
    client: LlamaCloud,
    pipeline_id: Optional[str],
    project: Optional[Project],
    pipeline_name: Optional[str],
) -> Pipeline:
    if pipeline_id is not None:
        return client.pipelines.get_pipeline(pipeline_id=pipeline_id)
    else:
        pipelines = client.pipelines.search_pipelines(
            project_id=project.id,
            pipeline_name=pipeline_name,
            pipeline_type=PipelineType.MANAGED.value,
        )
        if len(pipelines) == 0:
            raise ValueError(
                f"Unknown index name {pipeline_name}. Please confirm an index with this name exists."
            )
        elif len(pipelines) > 1:
            raise ValueError(
                f"Multiple pipelines found with name {pipeline_name} in project {project.name}"
            )
        return pipelines[0]


def resolve_project_and_pipeline(
    client: LlamaCloud,
    pipeline_name: Optional[str],
    pipeline_id: Optional[str],
    project_name: Optional[str],
    project_id: Optional[str],
    organization_id: Optional[str],
) -> Tuple[Project, Pipeline]:
    # resolve pipeline by ID
    if pipeline_id is not None:
        pipeline = resolve_pipeline(
            client, pipeline_id=pipeline_id, project=None, pipeline_name=None
        )
        project_id = pipeline.project_id

    # resolve project
    project = resolve_project(client, project_name, project_id, organization_id)

    # resolve pipeline by name
    if pipeline_id is None:
        pipeline = resolve_pipeline(
            client, pipeline_id=None, project=project, pipeline_name=pipeline_name
        )

    return project, pipeline
