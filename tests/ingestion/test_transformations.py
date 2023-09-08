from typing import Type
import pytest
from llama_index.schema import BaseComponent
from llama_index.ingestion.transformations import (
    ALL_COMPONENTS,
    PipelineTransformation,
    get_pipeline_transformation_from_serialized_component,
)

from llama_index.node_parser import SimpleNodeParser


@pytest.mark.skip
@pytest.mark.parametrize("component_type", ALL_COMPONENTS)
def test_can_generate_schema_for_pipeline_transforms_parametrized(
    component_type: Type[BaseComponent],
) -> None:
    class_schema = PipelineTransformation[component_type].schema()
    assert class_schema is not None
    assert len(class_schema) > 0


def test_can_build_pipeline_transform_from_serialized_component() -> None:
    serialized_parser = SimpleNodeParser.from_defaults().to_dict()
    pipeline_transformation = get_pipeline_transformation_from_serialized_component(
        serialized_parser
    )
    assert isinstance(pipeline_transformation, PipelineTransformation[SimpleNodeParser])
