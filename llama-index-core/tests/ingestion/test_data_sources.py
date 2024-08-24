import pytest
from llama_index.core.ingestion.data_sources import (
    ConfigurableDataSources,
    ConfiguredDataSource,
)
from llama_index.core.schema import Document


@pytest.mark.parametrize("configurable_data_source_type", ConfigurableDataSources)
def test_can_generate_schema_for_data_source_component_type(
    configurable_data_source_type: ConfigurableDataSources,
) -> None:
    schema = configurable_data_source_type.value.model_json_schema()  # type: ignore
    assert schema is not None
    assert len(schema) > 0

    # also check that we can generate schemas for
    # ConfiguredDataSource[component_type]
    component_type = configurable_data_source_type.value.component_type
    configured_schema = ConfiguredDataSource[component_type].model_json_schema()  # type: ignore
    assert configured_schema is not None
    assert len(configured_schema) > 0


def test_can_build_configured_data_source_from_component() -> None:
    document = Document.example()
    configured_data_source = ConfiguredDataSource.from_component(document)
    assert isinstance(
        configured_data_source,
        ConfiguredDataSource[Document],  # type: ignore
    )
    assert (
        configured_data_source.configurable_data_source_type.value.component_type
        == Document
    )


def test_build_configured_data_source() -> None:
    document = Document.example()
    configured_data_source = (
        ConfigurableDataSources.DOCUMENT.build_configured_data_source(document)
    )
    assert isinstance(
        configured_data_source,
        ConfiguredDataSource[Document],  # type: ignore
    )


def test_unique_configurable_data_source_names() -> None:
    names = set()
    for configurable_data_source_type in ConfigurableDataSources:
        assert configurable_data_source_type.value.name not in names
        names.add(configurable_data_source_type.value.name)
    assert len(names) > 0
    assert len(names) == len(ConfigurableDataSources)
