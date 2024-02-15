from enum import Enum
from typing import Generic, Type, TypeVar

from llama_index.core.bridge.pydantic import BaseModel, Field, GenericModel
from llama_index.core.vector_stores.types import BasePydanticVectorStore


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


def build_conifurable_data_sink_enum():
    """
    Build an enum of configurable data sinks.
    But conditional on if the corresponding vector store is available.
    """

    class ConfigurableComponent(Enum):
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

    enum_members = []

    try:
        from llama_index.vector_stores.chroma import (
            ChromaVectorStore,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "CHROMA",
                DataSink(
                    name="Chroma",
                    component_type=ChromaVectorStore,
                ),
            )
        )
    except ImportError:
        pass

    try:
        from llama_index.vector_stores.pinecone import (
            PineconeVectorStore,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "PINECONE",
                DataSink(
                    name="Pinecone",
                    component_type=PineconeVectorStore,
                ),
            )
        )
    except ImportError:
        pass

    try:
        from llama_index.vector_stores.postgres import (
            PGVectorStore,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "POSTGRES",
                DataSink(
                    name="PostgreSQL",
                    component_type=PGVectorStore,
                ),
            )
        )
    except ImportError:
        pass

    try:
        from llama_index.vector_stores.qdrant import (
            QdrantVectorStore,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "QDRANT",
                DataSink(
                    name="Qdrant",
                    component_type=QdrantVectorStore,
                ),
            )
        )
    except ImportError:
        pass

    try:
        from llama_index.vector_stores.weaviate import (
            WeaviateVectorStore,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "WEAVIATE",
                DataSink(
                    name="Weaviate",
                    component_type=WeaviateVectorStore,
                ),
            )
        )
    except ImportError:
        pass

    return ConfigurableComponent("ConfigurableDataSinks", enum_members)


ConfigurableDataSinks = build_conifurable_data_sink_enum()


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
