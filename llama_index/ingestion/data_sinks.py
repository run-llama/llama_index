from enum import Enum
from typing import Generic, Type, TypeVar

from llama_index.bridge.pydantic import BaseModel, Field, GenericModel
from llama_index.vector_stores import (
    ChromaVectorStore,
    PGVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    WeaviateVectorStore,
)
from llama_index.vector_stores.types import BasePydanticVectorStore


class DataSink(BaseModel):
    """
    A class containing metadata for a type of data sink.
    """

    name: str = Field(
        description="Unique and human-readable name for the type of data sink"
    )
    component_type: Type[BasePydanticVectorStore] = Field(
        description="Type of component that implements the data sink"
    )


class ConfigurableDataSinks(Enum):
    """
    Enumeration of all supported DataSink instances.
    """

    CHROMA = DataSink(
        name="Chroma",
        component_type=ChromaVectorStore,
    )

    PINECONE = DataSink(
        name="Pinecone",
        component_type=PineconeVectorStore,
    )

    POSTGRES = DataSink(
        name="PostgreSQL",
        component_type=PGVectorStore,
    )

    QDRANT = DataSink(
        name="Qdrant",
        component_type=QdrantVectorStore,
    )

    WEAVIATE = DataSink(
        name="Weaviate",
        component_type=WeaviateVectorStore,
    )

    @classmethod
    def from_component(
        cls, component: BasePydanticVectorStore
    ) -> "ConfigurableDataSinks":
        component_class = type(component)
        for component_type in cls:
            if component_type.value.component_type == component_class:
                return component_type
        raise ValueError(
            f"Component {component} is not a supported data sink component."
        )

    def build_configured_data_sink(
        self, component: BasePydanticVectorStore
    ) -> "ConfiguredDataSink":
        component_type = self.value.component_type
        if not isinstance(component, component_type):
            raise ValueError(
                f"The enum value {self} is not compatible with component of "
                f"type {type(component)}"
            )
        return ConfiguredDataSink[component_type](  # type: ignore
            component=component, name=self.value.name
        )


T = TypeVar("T", bound=BasePydanticVectorStore)


class ConfiguredDataSink(GenericModel, Generic[T]):
    """
    A class containing metadata & implementation for a data sink in a pipeline.
    """

    name: str
    component: T = Field(description="Component that implements the data sink")

    @classmethod
    def from_component(cls, component: BasePydanticVectorStore) -> "ConfiguredDataSink":
        """
        Build a ConfiguredDataSink from a component.
        This should be the preferred way to build a ConfiguredDataSink
        as it will ensure that the component is supported as indicated by having a
        corresponding enum value in DataSources.
        This has the added bonus that you don't need to specify the generic type
        like ConfiguredDataSink[Document]. The return value of
        this ConfiguredDataSink.from_component(document) will be
        ConfiguredDataSink[Document] if document is
        a Document object.
        """
        return ConfigurableDataSinks.from_component(
            component
        ).build_configured_data_sink(component)

    @property
    def configurable_data_sink_type(self) -> ConfigurableDataSinks:
        return ConfigurableDataSinks.from_component(self.component)
