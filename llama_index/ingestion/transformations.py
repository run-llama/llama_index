"""
This module maintains the list of transformations that are supported by the system.
"""

from typing import Dict, Sequence, Set, Type, TypeVar, Generic
from enum import Enum

from llama_index.bridge.pydantic import BaseModel, Field, GenericModel
from llama_index.schema import Document, BaseNode, BaseComponent
from llama_index.node_parser.extractors import (
    BaseExtractor,
    MetadataExtractor,
    KeywordExtractor,
    TitleExtractor,
    EntityExtractor,
    MarvinMetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.node_parser import (
    NodeParser,
    SimpleNodeParser,
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
)


# Transform Input/Output Types
class TransformationIOType(BaseModel):
    name: str = Field(description="Name of the input/output type")
    description: str = Field(description="Description of the input/output type")
    python_type: str = Field(description="Python type of the input/output type")


class TransformationIOTypes(Enum):
    DOCUMENTS = TransformationIOType(
        name="Documents",
        description="Documents",
        python_type=str(Sequence[Document]),
    )
    NODES = TransformationIOType(
        name="Nodes",
        description="Nodes",
        python_type=str(Sequence[BaseNode]),
    )


class TransformationType(BaseModel):
    """A description for a type of transformation within a pipeline."""

    name: str = Field(description="Unique name of the type of transformation")
    description: str = Field(description="Description for the type of transformation")
    input_type: TransformationIOType = Field(
        description="Input type for the transformation type"
    )
    output_type: TransformationIOType = Field(
        description="Output type for the transformation type"
    )


class TransformationTypes(Enum):
    """Supported transformations."""

    METADATA_EXTRACTOR = TransformationType(
        name="MetadataExtractor",
        description="Applies a function to extract metadata from nodes",
        input_type=TransformationIOTypes.NODES.value,
        output_type=TransformationIOTypes.NODES.value,
    )
    NODE_PARSER = TransformationType(
        name="NodeParser",
        description="Applies a function to parse nodes from documents",
        input_type=TransformationIOTypes.DOCUMENTS.value,
        output_type=TransformationIOTypes.NODES.value,
    )


# Keep this up-to-date with the set of supported MetadataExtractors
METADATA_EXTRACTOR_COMPONENTS: Set[Type[BaseExtractor]] = {
    MetadataExtractor,
    KeywordExtractor,
    TitleExtractor,
    EntityExtractor,
    MarvinMetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
}

# Keep this up-to-date with the set of supported NodeParsers
NODE_PARSER_COMPONENTS: Set[Type[NodeParser]] = {
    SimpleNodeParser,
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
}

_component_type_to_transformation_type: Dict[
    Type[BaseComponent], TransformationTypes
] = {
    **{
        component_type: TransformationTypes.METADATA_EXTRACTOR
        for component_type in METADATA_EXTRACTOR_COMPONENTS
    },
    **{
        component_type: TransformationTypes.NODE_PARSER
        for component_type in NODE_PARSER_COMPONENTS
    },
}

ALL_COMPONENTS: Set[Type[BaseComponent]] = {
    *METADATA_EXTRACTOR_COMPONENTS,
    *NODE_PARSER_COMPONENTS,
}

T = TypeVar("T", bound=BaseComponent)


class PipelineTransformation(GenericModel, Generic[T]):
    """
    A class containing the metdata + implementation for a transformation within a pipeline.
    """

    transformation_type: TransformationTypes = Field(
        description="Type of transformation"
    )
    component: T = Field(description="Component that implements the transformation")

    @classmethod
    def from_component(cls, component: T) -> "PipelineTransformation[T]":
        component_class = cls.__fields__["component"].type_
        if not isinstance(component, component_class):
            raise ValueError(
                "Given component is of a different type than the requested "
                "PipelineTransformation[T]'s component type."
            )

        transformation_type: TransformationTypes = (
            _component_type_to_transformation_type[component_class]
        )

        return cls(
            transformation_type=transformation_type,
            component=component,
        )
