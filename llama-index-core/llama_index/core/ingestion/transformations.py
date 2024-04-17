"""
This module maintains the list of transformations that are supported by the system.
"""

from enum import Enum
from typing import Generic, Sequence, Type, TypeVar

from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    GenericModel,
    ValidationError,
)
from llama_index.core.node_parser import (
    CodeSplitter,
    HTMLNodeParser,
    JSONNodeParser,
    MarkdownNodeParser,
    SentenceSplitter,
    SimpleFileNodeParser,
    TokenTextSplitter,
    MarkdownElementNodeParser,
)
from llama_index.core.schema import BaseComponent, BaseNode, Document


# Transform Input/Output Types
class TransformationIOType(BaseModel):
    name: str = Field(description="Name of the input/output type")
    description: str = Field(description="Description of the input/output type")
    python_type: str = Field(description="Python type of the input/output type")


class TransformationIOTypes(Enum):
    DOCUMENTS = TransformationIOType(
        name="Documents",
        description="A sequence of Documents",
        python_type=str(Sequence[Document]),
    )
    NODES = TransformationIOType(
        name="Nodes",
        description="A sequence of Nodes from a sequence of Documents",
        python_type=str(Sequence[BaseNode]),
    )


class TransformationCategory(BaseModel):
    """A description for a category of transformation within a pipeline."""

    name: str = Field(description="Unique name of the type of transformation")
    description: str = Field(description="Description for the type of transformation")
    input_type: TransformationIOType = Field(
        description="Input type for the transformation type"
    )
    output_type: TransformationIOType = Field(
        description="Output type for the transformation type"
    )


class TransformationCategories(Enum):
    """Supported transformation categories."""

    NODE_PARSER = TransformationCategory(
        name="NodeParser",
        description="Applies a function to parse nodes from documents",
        input_type=TransformationIOTypes.DOCUMENTS.value,
        output_type=TransformationIOTypes.NODES.value,
    )
    EMBEDDING = TransformationCategory(
        name="Embedding",
        description="Applies a function to embed nodes",
        input_type=TransformationIOTypes.NODES.value,
        output_type=TransformationIOTypes.NODES.value,
    )


class ConfigurableTransformation(BaseModel):
    """
    A class containing metadata for a type of transformation that can be in a pipeline.
    """

    name: str = Field(
        description="Unique and human-readable name for the type of transformation"
    )
    transformation_category: TransformationCategories = Field(
        description="Type of transformation"
    )
    component_type: Type[BaseComponent] = Field(
        description="Type of component that implements the transformation"
    )


def build_configurable_transformation_enum():
    """
    Build an enum of configurable transformations.
    But conditional on if the corresponding component is available.
    """

    class ConfigurableComponent(Enum):
        @classmethod
        def from_component(
            cls, component: BaseComponent
        ) -> "ConfigurableTransformations":
            component_class = type(component)
            for component_type in cls:
                if component_type.value.component_type == component_class:
                    return component_type
            raise ValueError(
                f"Component {component} is not a supported transformation component."
            )

        def build_configured_transformation(
            self, component: BaseComponent
        ) -> "ConfiguredTransformation":
            component_type = self.value.component_type
            if not isinstance(component, component_type):
                raise ValueError(
                    f"The enum value {self} is not compatible with component of "
                    f"type {type(component)}"
                )
            return ConfiguredTransformation[component_type](  # type: ignore
                component=component, name=self.value.name
            )

    enum_members = []

    # Node parsers
    enum_members.append(
        (
            "CODE_NODE_PARSER",
            ConfigurableTransformation(
                name="Code Splitter",
                transformation_category=TransformationCategories.NODE_PARSER,
                component_type=CodeSplitter,
            ),
        )
    )

    enum_members.append(
        (
            "SENTENCE_AWARE_NODE_PARSER",
            ConfigurableTransformation(
                name="Sentence Splitter",
                transformation_category=TransformationCategories.NODE_PARSER,
                component_type=SentenceSplitter,
            ),
        )
    )

    enum_members.append(
        (
            "TOKEN_AWARE_NODE_PARSER",
            ConfigurableTransformation(
                name="Token Text Splitter",
                transformation_category=TransformationCategories.NODE_PARSER,
                component_type=TokenTextSplitter,
            ),
        )
    )

    enum_members.append(
        (
            "HTML_NODE_PARSER",
            ConfigurableTransformation(
                name="HTML Node Parser",
                transformation_category=TransformationCategories.NODE_PARSER,
                component_type=HTMLNodeParser,
            ),
        )
    )

    enum_members.append(
        (
            "MARKDOWN_NODE_PARSER",
            ConfigurableTransformation(
                name="Markdown Node Parser",
                transformation_category=TransformationCategories.NODE_PARSER,
                component_type=MarkdownNodeParser,
            ),
        )
    )

    enum_members.append(
        (
            "JSON_NODE_PARSER",
            ConfigurableTransformation(
                name="JSON Node Parser",
                transformation_category=TransformationCategories.NODE_PARSER,
                component_type=JSONNodeParser,
            ),
        )
    )

    enum_members.append(
        (
            "SIMPLE_FILE_NODE_PARSER",
            ConfigurableTransformation(
                name="Simple File Node Parser",
                transformation_category=TransformationCategories.NODE_PARSER,
                component_type=SimpleFileNodeParser,
            ),
        )
    )

    enum_members.append(
        (
            "MARKDOWN_ELEMENT_NODE_PARSER",
            ConfigurableTransformation(
                name="Markdown Element Node Parser",
                transformation_category=TransformationCategories.NODE_PARSER,
                component_type=MarkdownElementNodeParser,
            ),
        )
    )

    # Embeddings
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding  # pants: no-infer-dep

        enum_members.append(
            (
                "OPENAI_EMBEDDING",
                ConfigurableTransformation(
                    name="OpenAI Embedding",
                    transformation_category=TransformationCategories.EMBEDDING,
                    component_type=OpenAIEmbedding,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.embeddings.azure_openai import (
            AzureOpenAIEmbedding,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "AZURE_EMBEDDING",
                ConfigurableTransformation(
                    name="Azure OpenAI Embedding",
                    transformation_category=TransformationCategories.EMBEDDING,
                    component_type=AzureOpenAIEmbedding,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.embeddings.huggingface import (
            HuggingFaceInferenceAPIEmbedding,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "HUGGINGFACE_API_EMBEDDING",
                ConfigurableTransformation(
                    name="HuggingFace API Embedding",
                    transformation_category=TransformationCategories.EMBEDDING,
                    component_type=HuggingFaceInferenceAPIEmbedding,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    return ConfigurableComponent("ConfigurableTransformations", enum_members)


ConfigurableTransformations = build_configurable_transformation_enum()

T = TypeVar("T", bound=BaseComponent)


class ConfiguredTransformation(GenericModel, Generic[T]):
    """
    A class containing metadata & implementation for a transformation in a pipeline.
    """

    name: str
    component: T = Field(description="Component that implements the transformation")

    @classmethod
    def from_component(cls, component: BaseComponent) -> "ConfiguredTransformation":
        """
        Build a ConfiguredTransformation from a component.

        This should be the preferred way to build a ConfiguredTransformation
        as it will ensure that the component is supported as indicated by having a
        corresponding enum value in ConfigurableTransformations.

        This has the added bonus that you don't need to specify the generic type
        like ConfiguredTransformation[SentenceSplitter]. The return value of
        this ConfiguredTransformation.from_component(simple_node_parser) will be
        ConfiguredTransformation[SentenceSplitter] if simple_node_parser is
        a SentenceSplitter.
        """
        return ConfigurableTransformations.from_component(
            component
        ).build_configured_transformation(component)

    @property
    def configurable_transformation_type(self) -> ConfigurableTransformations:
        return ConfigurableTransformations.from_component(self.component)
