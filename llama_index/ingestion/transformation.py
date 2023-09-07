"""
This module maintains the list of transformations that are supported by the system.
"""

from typing import Any, Dict
from enum import Enum

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    KeywordExtractor,
    TitleExtractor,
    EntityExtractor,
    MarvinMetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.node_parser import (
    SimpleNodeParser,
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
)


# Transform Input/Output Types
class TransformationIOType(BaseModel):
    name: str = Field(description="Name of the input/output type")
    description: str = Field(description="Description of the input/output type")
    python_type: str = Field(description="Python type of the input/output type")


# TODO: Figure out how to do this with an Enum class
class TransformationIOTypes(Enum):
    """Input/Output types for transformations."""

    DOCUMENTS = TransformationIOType(
        name="Documents",
        description="Documents",
        python_type="Sequence[Document]",
    )
    NODES = TransformationIOType(
        name="Nodes",
        description="Nodes",
        python_type="Sequence[BaseNode]",
    )


# Configured transformation schemas
class ConfiguredTransformation(BaseModel):
    """A transformation that can be applied to data."""

    name: str = Field(description="Unique name of the transformation")
    description: str = Field(description="Description for the transformation")
    input_type: TransformationIOType = Field(
        description="Input type for the transformation"
    )
    output_type: TransformationIOType = Field(
        description="Output type for the transformation"
    )
    configuration_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="The configurable fields for the transformation",
    )


# TODO: Figure out how to do this with an Enum class
class SupportedTransformations(Enum):
    """Supported transformations."""

    METADATA_EXTRACTOR = ConfiguredTransformation(
        name="MetadataExtractor",
        description="Applies a function to extract metadata from nodes",
        input_type=TransformationIOTypes.NODES.value,
        output_type=TransformationIOTypes.NODES.value,
    )
    NODE_PARSER = ConfiguredTransformation(
        name="NodeParser",
        description="Applies a function to parse nodes from documents",
        input_type=TransformationIOTypes.DOCUMENTS.value,
        output_type=TransformationIOTypes.NODES.value,
    )


# Class name to transformation utilities
CLASS_NAME_TO_TRANSFORM: Dict[str, ConfiguredTransformation] = {
    # Metadata Extractors
    MetadataExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR.value,
    KeywordExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR.value,
    TitleExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR.value,
    EntityExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR.value,
    MarvinMetadataExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR.value,  # noqa: E501
    SummaryExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR.value,
    QuestionsAnsweredExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR.value,  # noqa: E501
    # Node Parsers
    SimpleNodeParser.class_name(): SupportedTransformations.NODE_PARSER.value,
    SentenceWindowNodeParser.class_name(): SupportedTransformations.NODE_PARSER.value,
    HierarchicalNodeParser.class_name(): SupportedTransformations.NODE_PARSER.value,
}


def get_configured_transform(
    transform_schema: Dict[str, Any]
) -> ConfiguredTransformation:
    class_name = transform_schema.pop("class_name", None)
    if class_name is None:
        raise ValueError(
            "transform_schema must have a class_name field. Current input is invalid."
        )

    if class_name not in CLASS_NAME_TO_TRANSFORM:
        raise ValueError(
            f"transform_schema has an invalid class_name field: {class_name}. "
            f"Current transform is not supported."
        )

    configured_transform = CLASS_NAME_TO_TRANSFORM[class_name]
    configured_transform.configuration_schema = transform_schema
    return configured_transform
