from enum import Enum
from typing import Generic, Type, TypeVar

from llama_index.bridge.pydantic import BaseModel, Field, GenericModel
from llama_index.readers import ReaderConfig
from llama_index.schema import BaseComponent, Document


class RawFile(BaseComponent):
    """A raw file."""

    file_bytes: bytes = Field(description="The raw file bytes.")

    @classmethod
    def from_path(cls, path: str) -> "RawFile":
        """Load a raw file from a path."""
        with open(path, "rb") as f:
            return cls(file_bytes=f.read())


class DataSource(BaseModel):
    """
    A class containing metdata for a type of data source
    """

    name: str = Field(
        description="Unique and human-readable name for the type of data source"
    )
    component_type: Type[BaseComponent] = Field(
        description="Type of component that implements the data source"
    )


class ConfigurableDataSources(Enum):
    """
    Enumeration of all supported DataSource instances.
    """

    DOCUMENT = DataSource(
        name="Document",
        component_type=Document,
    )

    READER = DataSource(
        name="Reader",
        component_type=ReaderConfig,
    )

    RAW_FILE = DataSource(
        name="Raw File",
        component_type=RawFile,
    )

    @classmethod
    def from_component(cls, component: BaseComponent) -> "ConfigurableDataSources":
        component_class = type(component)
        for component_type in cls:
            if component_type.value.component_type == component_class:
                return component_type
        raise ValueError(
            f"Component {component} is not a supported data source component."
        )

    def build_configured_data_source(
        self, component: BaseComponent
    ) -> "ConfigurableDataSources":
        component_type = self.value.component_type
        if not isinstance(component, component_type):
            raise ValueError(
                f"The enum value {self} is not compatible with component of "
                f"type {type(component)}"
            )
        return ConfiguredDataSource[component_type](component=component)  # type: ignore


T = TypeVar("T", bound=BaseComponent)


class ConfiguredDataSource(GenericModel, Generic[T]):
    """
    A class containing metdata & implementation for a data source in a pipeline.
    """

    component: T = Field(description="Component that implements the data source")

    @classmethod
    def from_component(cls, component: BaseComponent) -> "ConfiguredDataSource":
        """
        Build a ConfiguredDataSource from a component.

        This should be the preferred way to build a ConfiguredDataSource
        as it will ensure that the component is supported as indicated by having a
        corresponding enum value in DataSources.

        This has the added bonus that you don't need to specify the generic type
        like ConfiguredDataSource[Document]. The return value of
        this ConfiguredDataSource.from_component(document) will be
        ConfiguredDataSource[Document] if document is
        a Document object.
        """
        return ConfigurableDataSources.from_component(
            component
        ).build_configured_data_source(component)

    @property
    def configurable_data_source_type(self) -> ConfigurableDataSources:
        return ConfigurableDataSources.from_component(self.component)
