"""
This module maintains the list of transformations that are supported by dashscope.
"""

from enum import Enum
from typing import Generic, TypeVar

from llama_index.core.bridge.pydantic import (
    Field,
    BaseModel,
    ValidationError,
)

from llama_index.core.schema import BaseComponent
from llama_index.core.ingestion.transformations import (
    TransformationCategories,
    ConfigurableTransformation,
)


def dashscope_build_configurable_transformation_enum():
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
        ) -> "DashScopeConfiguredTransformation":
            component_type = self.value.component_type
            if not isinstance(component, component_type):
                raise ValueError(
                    f"The enum value {self} is not compatible with component of "
                    f"type {type(component)}"
                )
            return DashScopeConfiguredTransformation[component_type](  # type: ignore
                component=component, name=self.value.name
            )

    enum_members = []

    # Node parsers
    try:
        from llama_index.node_parser.dashscope import DashScopeJsonNodeParser

        enum_members.append(
            (
                "DASHSCOPE_JSON_NODE_PARSER",
                ConfigurableTransformation(
                    name="DashScope Json Node Parser",
                    transformation_category=TransformationCategories.NODE_PARSER,
                    component_type=DashScopeJsonNodeParser,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    # Embeddings
    try:
        from llama_index.embeddings.dashscope import (
            DashScopeEmbedding,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "DASHSCOPE_EMBEDDING",
                ConfigurableTransformation(
                    name="DashScope Embedding",
                    transformation_category=TransformationCategories.EMBEDDING,
                    component_type=DashScopeEmbedding,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    return ConfigurableComponent("ConfigurableTransformations", enum_members)


ConfigurableTransformations = dashscope_build_configurable_transformation_enum()

T = TypeVar("T", bound=BaseComponent)


class DashScopeConfiguredTransformation(BaseModel, Generic[T]):
    """
    A class containing metadata & implementation for a transformation in a dashscope pipeline.
    """

    name: str
    component: T = Field(description="Component that implements the transformation")

    @classmethod
    def from_component(cls, component: BaseComponent) -> "ConfiguredTransformation":
        """
        Build a ConfiguredTransformation from a component in dashscope.
        """
        return ConfigurableTransformations.from_component(
            component
        ).build_configured_transformation(component)

    @property
    def configurable_transformation_type(self) -> ConfigurableTransformations:
        return ConfigurableTransformations.from_component(self.component)
