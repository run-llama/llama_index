import sys
from unittest.mock import MagicMock

import pytest
from llama_index.core.ingestion.data_sinks import (
    ConfigurableDataSinks,
    ConfiguredDataSink,
)

try:
    from llama_index.vector_store.weaviate import WeaviateVectorStore
except ImportError:
    WeaviateVectorStore = None


@pytest.mark.parametrize("configurable_data_sink_type", ConfigurableDataSinks)
def test_can_generate_schema_for_data_sink_component_type(
    configurable_data_sink_type: ConfigurableDataSinks,
) -> None:
    schema = configurable_data_sink_type.value.model_json_schema()  # type: ignore
    assert schema is not None
    assert len(schema) > 0

    # also check that we can generate schemas for
    # ConfiguredDataSink[component_type]
    component_type = configurable_data_sink_type.value.component_type
    configured_schema = ConfiguredDataSink[component_type].model_json_schema()  # type: ignore
    assert configured_schema is not None
    assert len(configured_schema) > 0


@pytest.mark.skipif(WeaviateVectorStore is None, reason="weaviate not installed")
def test_can_build_configured_data_sink_from_component() -> None:
    sys.modules["weaviate"] = MagicMock()
    weaviate_client = MagicMock()
    batch_context_manager = MagicMock()
    weaviate_client.batch.__enter__.return_value = batch_context_manager

    vector_store = WeaviateVectorStore(weaviate_client=weaviate_client)
    configured_data_sink = ConfiguredDataSink.from_component(vector_store)
    assert isinstance(
        configured_data_sink,
        ConfiguredDataSink[WeaviateVectorStore],  # type: ignore
    )
    assert (
        configured_data_sink.configurable_data_sink_type.value.component_type
        == WeaviateVectorStore
    )


@pytest.mark.skipif(WeaviateVectorStore is None, reason="weaviate not installed")
def test_build_configured_data_sink() -> None:
    sys.modules["weaviate"] = MagicMock()
    weaviate_client = MagicMock()
    batch_context_manager = MagicMock()
    weaviate_client.batch.__enter__.return_value = batch_context_manager

    vector_store = WeaviateVectorStore(weaviate_client=weaviate_client)
    configured_data_sink = ConfigurableDataSinks.WEAVIATE.build_configured_data_sink(
        vector_store
    )
    assert isinstance(
        configured_data_sink,
        ConfiguredDataSink[WeaviateVectorStore],  # type: ignore
    )

    with pytest.raises(ValueError):
        ConfigurableDataSinks.PINECONE.build_configured_data_sink(vector_store)


def test_unique_configurable_data_sink_names() -> None:
    names = set()
    for configurable_data_sink_type in ConfigurableDataSinks:
        assert configurable_data_sink_type.value.name not in names
        names.add(configurable_data_sink_type.value.name)
    assert len(names) == len(ConfigurableDataSinks)
