"""
This module maintains the list of transformations that are supported by the system.
"""

from typing import Any, Dict
from enum import Enum
from pydantic import BaseModel, Field


class TransformationIOType(BaseModel):
    name: str = Field(description="Name of the input/output type")
    description: str = Field(description="Description of the input/output type")
    python_type: str = Field(description="Python type of the input/output type")


TransformationIOTypes = Enum(
    value="TransformationIOTypes",
    names=[
        (
            "DOCUMENTS",
            TransformationIOType(
                name="Documents",
                description="Documents",
                python_type="Sequence[Document]",
            ),
        ),
        (
            "NODES",
            TransformationIOType(
                name="Nodes",
                description="Nodes",
                python_type="Sequence[BaseNode]",
            ),
        ),
        (
            "TEXT",
            TransformationIOType(
                name="Text",
                description="Text",
                python_type="str",
            ),
        ),
        (
            "TEXT_LIST",
            TransformationIOType(
                name="TextList",
                description="TextList",
                python_type="Sequence[str]",
            ),
        ),
    ],
)


class ConfiguredTransformation(BaseModel):
    """A transformation that can be applied to data."""

    name: str = Field(description="Unique name of the transformation")
    description: str = Field(description="Description for the transformation")
    input_type: TransformationIOTypes = Field(
        description="Input type for the transformation"
    )
    output_type: TransformationIOTypes = Field(
        description="Output type for the transformation"
    )
    configuration_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="The configurable fields for the transformation",
    )


SupportedTransformations = Enum(
    value="ConfiguredTransformation",
    names=[
        (
            "METADATA_EXTRACTOR",
            ConfiguredTransformation(
                name="MetadataExtractor",
                description="Applies a function to extract metadata from nodes",
                input_type=TransformationIOTypes.NODES,
                output_type=TransformationIOTypes.NODES,
            ),
        ),
        (
            "NODE_PARSER",
            ConfiguredTransformation(
                name="NodeParser",
                description="Applies a function to parse nodes from documents",
                input_type=TransformationIOTypes.DOCUMENTS,
                output_type=TransformationIOTypes.NODES,
            ),
        ),
        (
            "TEXT_SPLITTER",
            ConfiguredTransformation(
                name="TextSplitter",
                description=(
                    "Applies a function to split text into chunks, "
                    "used in a NodeParser"
                ),
                input_type=TransformationIOTypes.TEXT,
                output_type=TransformationIOTypes.TEXT_LIST,
            ),
        ),
    ],
)


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
from llama_index.text_splitter import SentenceSplitter, TokenTextSplitter, CodeSplitter


CLASS_NAME_TO_TRANSFORM: Dict[str, ConfiguredTransformation] = {
    # Metadata Extractors
    MetadataExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR,
    KeywordExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR,
    TitleExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR,
    EntityExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR,
    MarvinMetadataExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR,
    SummaryExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR,
    QuestionsAnsweredExtractor.class_name(): SupportedTransformations.METADATA_EXTRACTOR,
    # Node Parsers
    SimpleNodeParser.class_name(): SupportedTransformations.NODE_PARSER,
    SentenceWindowNodeParser.class_name(): SupportedTransformations.NODE_PARSER,
    HierarchicalNodeParser.class_name(): SupportedTransformations.NODE_PARSER,
    # Text Splitters
    SentenceSplitter.class_name(): SupportedTransformations.TEXT_SPLITTER,
    TokenTextSplitter.class_name(): SupportedTransformations.TEXT_SPLITTER,
    CodeSplitter.class_name(): SupportedTransformations.TEXT_SPLITTER,
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


simple_node_parser = SimpleNodeParser.from_defaults()
sentence_node_parser = SentenceWindowNodeParser.from_defaults()
hierarchical_node_parser = HierarchicalNodeParser.from_defaults()

keyword_extractor = KeywordExtractor()
metadata_extractor = MetadataExtractor(extractors=[keyword_extractor])

print(
    "SimpleNodeParser\n",
    get_configured_transform(simple_node_parser.to_dict()).configuration_schema,
)
print("------")
print(
    "SentenceWindowNodeParser\n",
    get_configured_transform(sentence_node_parser.to_dict()).configuration_schema,
)
print("------")
print(
    "HierarchicalNodeParser\n",
    get_configured_transform(hierarchical_node_parser.to_dict()).configuration_schema,
)
print("------")
print(
    "KeywordExtractor\n",
    get_configured_transform(keyword_extractor.to_dict()).configuration_schema,
)
print("------")
print(
    "MetadataExtractor\n",
    get_configured_transform(metadata_extractor.to_dict()).configuration_schema,
)
