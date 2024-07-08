import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.box import BoxReader

from box_sdk_gen import BoxClient


from tests.conftest import get_testing_data


def test_class():
    names_of_base_classes = [b.__name__ for b in BoxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_reader_init(box_client_ccg_unit_testing: BoxClient):
    reader = BoxReader(box_client=box_client_ccg_unit_testing)

    # schema = reader.schema()
    # assert schema is not None
    # assert len(schema) > 0
    # assert "box_client_id" in schema["properties"]

    # json = reader.json(exclude_unset=True)

    # new_reader = BoxReader.parse_raw(json)
    # assert new_reader is not None
    # assert new_reader.box_client_id == reader.box_client_id
    # assert new_reader.box_client_secret == reader.box_client_secret


####################################################################################################
# Integration tests
####################################################################################################


def test_box_reader_whoami(box_client_ccg_integration_testing: BoxClient):
    me = box_client_ccg_integration_testing.users.get_user_me()
    assert me is not None


def test_box_reader_csv(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()
    reader = BoxReader(box_client=box_client_ccg_integration_testing)
    docs = reader.load_data(file_ids=[test_data["test_csv_id"]])
    assert len(docs) == 1


def test_box_reader_folder(box_client_ccg_integration_testing):
    # Very slow test
    test_data = get_testing_data()
    if test_data["disable_slow_tests"]:
        raise pytest.skip(f"Slow integration tests are disabled.")
    reader = BoxReader(box_client=box_client_ccg_integration_testing)

    docs = reader.load_data(folder_id=test_data["test_folder_id"])
    assert len(docs) >= 1
