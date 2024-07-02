import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.box import BoxReader
from llama_index.readers.box.box_client_ccg import BoxConfigCCG


@pytest.fixture(scope="module")
def box_unit_testing_config():
    return BoxConfigCCG()


@pytest.fixture(scope="module")
def box_integration_testing_config(box_unit_testing_config):
    box_config = box_unit_testing_config
    if box_config.client_id == "YOUR_BOX_CLIENT_ID":
        raise pytest.skip(
            f"Create a .env file with the Box credentials to run integration testa."
        )
    return box_config


def test_class():
    names_of_base_classes = [b.__name__ for b in BoxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_serialize(box_unit_testing_config):
    box_config = box_unit_testing_config
    reader = BoxReader(
        box_client_id=box_config.client_id,
        box_client_secret=box_config.client_secret,
        box_enterprise_id=box_config.enterprise_id,
        box_user_id=box_config.ccg_user_id,
    )

    schema = reader.schema()
    assert schema is not None
    assert len(schema) > 0
    assert "box_client_id" in schema["properties"]

    json = reader.json(exclude_unset=True)

    new_reader = BoxReader.parse_raw(json)
    assert new_reader.box_client_id == reader.box_client_id
    assert new_reader.box_client_secret == reader.box_client_secret


####################################################################################################
# Integration tests
####################################################################################################


def test_box_reader_connect_config(box_integration_testing_config):
    box_config = box_integration_testing_config
    reader = BoxReader(
        box_client_id=box_config.client_id,
        box_client_secret=box_config.client_secret,
        box_enterprise_id=box_config.enterprise_id,
        box_user_id=box_config.ccg_user_id,
    )
    reader.load_data()


def test_mixins(box_integration_testing_config):
    box_config = box_integration_testing_config
    reader = BoxReader(
        box_client_id=box_config.client_id,
        box_client_secret=box_config.client_secret,
        box_enterprise_id=box_config.enterprise_id,
        box_user_id=box_config.ccg_user_id,
    )

    docs = reader.load_data()
    assert len(docs) > 0

    # resources = reader.list_resources()
    # assert len(resources) == len(docs)

    # resource = resources[0]
    # resource_info = reader.get_resource_info(resource)
    # assert resource_info is not None
    # assert resource_info["file_path"] == resource
    # assert resource_info["file_name"] in resource
    # assert resource_info["file_size"] > 0

    # file_content = reader.read_file_content(resource)
    # assert file_content is not None
    # assert len(file_content) == resource_info["file_size"]
