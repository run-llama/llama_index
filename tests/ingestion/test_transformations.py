from typing import Type
import pytest
from llama_index.schema import BaseComponent
from llama_index.ingestion.transformations import (
    ALL_COMPONENTS,
    PipelineTransformation,
    TransformationTypes,
)

from llama_index.node_parser import SimpleNodeParser, SentenceWindowNodeParser


@pytest.mark.parametrize("component_type", ALL_COMPONENTS)
def test_can_generate_schema_for_pipeline_transforms_parametrized(
    component_type: Type[BaseComponent],
) -> None:
    class_schema = PipelineTransformation[component_type].schema()  # type: ignore
    assert class_schema is not None
    assert len(class_schema) > 0


def test_can_build_pipeline_transform_from_serialized_component() -> None:
    parser = SimpleNodeParser.from_defaults()
    pipeline_transformation = PipelineTransformation[SimpleNodeParser].from_component(
        parser
    )
    assert isinstance(pipeline_transformation, PipelineTransformation[SimpleNodeParser])  # type: ignore
    assert not isinstance(pipeline_transformation, PipelineTransformation[SentenceWindowNodeParser])  # type: ignore
    assert (
        pipeline_transformation.transformation_type == TransformationTypes.NODE_PARSER
    )
