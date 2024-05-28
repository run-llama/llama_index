import json
from typing import List, Optional

from llama_index.indices.managed.dashscope.transformations import (
    DashScopeConfiguredTransformation,
)
from llama_index.core.schema import BaseNode, TransformComponent


def default_transformations() -> List[TransformComponent]:
    """Default transformations."""
    from llama_index.node_parser.dashscope import DashScopeJsonNodeParser
    from llama_index.embeddings.dashscope import (
        DashScopeEmbedding,
        DashScopeTextEmbeddingModels,
        DashScopeTextEmbeddingType,
    )

    node_parser = DashScopeJsonNodeParser()
    document_embedder = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    )
    return [
        node_parser,
        document_embedder,
    ]


def get_pipeline_create(
    name: str,
    transformations: Optional[List[TransformComponent]] = None,
    documents: Optional[List[BaseNode]] = None,
) -> dict:
    configured_transformations: List[DashScopeConfiguredTransformation] = []
    for transformation in transformations:
        try:
            configured_transformations.append(
                DashScopeConfiguredTransformation.from_component(transformation)
            )
        except ValueError:
            raise ValueError(f"Unsupported transformation: {type(transformation)}")

    configured_transformation_items: List[Dict] = []
    for item in configured_transformations:
        configured_transformation_items.append(
            {
                "component": json.loads(item.component.json()),
                "configurable_transformation_type": item.configurable_transformation_type.name,
            }
        )
    data_sources = [
        {
            "source_type": "DATA_CENTER_FILE",
            "component": {
                "doc_ids": [doc.node_id for doc in documents],
            },
        }
    ]
    return {
        "name": name,
        "pipeline_type": "MANAGED_SHARED",
        "configured_transformations": configured_transformation_items,
        "data_sources": data_sources,
        "data_sinks": [
            {
                "sink_type": "ES",
            }
        ],
        # for debug
        "data_type": "structured",
        "config_model": "recommend",
    }


def get_doc_insert(
    transformations: Optional[List[TransformComponent]] = None,
    documents: Optional[List[BaseNode]] = None,
) -> dict:
    configured_transformations: List[DashScopeConfiguredTransformation] = []
    for transformation in transformations:
        try:
            configured_transformations.append(
                DashScopeConfiguredTransformation.from_component(transformation)
            )
        except ValueError:
            raise ValueError(f"Unsupported transformation: {type(transformation)}")

    configured_transformation_items: List[Dict] = []
    for item in configured_transformations:
        configured_transformation_items.append(
            {
                "component": json.loads(item.component.json()),
                "configurable_transformation_type": item.configurable_transformation_type.name,
            }
        )
    data_sources = [
        {
            "source_type": "DATA_CENTER_FILE",
            "component": {
                "doc_ids": [doc.node_id for doc in documents],
            },
        }
    ]
    return {
        "configured_transformations": configured_transformation_items,
        "data_sources": data_sources,
    }


def get_doc_delete(ref_doc_ids: List[str]) -> dict:
    data_sources = [
        {
            "source_type": "DATA_CENTER_FILE",
            "component": {
                "doc_ids": ref_doc_ids,
            },
        }
    ]
    return {
        "data_sources": data_sources,
    }
