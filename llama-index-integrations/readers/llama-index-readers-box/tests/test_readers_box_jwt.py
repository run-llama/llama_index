import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.box import BoxReader

from box_sdk_gen import BoxClient

from tests.conftest import get_testing_data


def test_class():
    names_of_base_classes = [b.__name__ for b in BoxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_reader_init(box_client_jwt_unit_testing: BoxClient):
    reader = BoxReader(box_client=box_client_jwt_unit_testing)


####################################################################################################
# Integration tests
####################################################################################################


def test_box_reader_whoami(box_client_jwt_integration_testing: BoxClient):
    me = box_client_jwt_integration_testing.users.get_user_me()
    assert me is not None


def test_box_reader_csv(box_client_jwt_integration_testing: BoxClient):
    test_data = get_testing_data()
    reader = BoxReader(box_client=box_client_jwt_integration_testing)
    docs = reader.load_data(file_ids=[test_data["test_csv_id"]])
    assert len(docs) == 1


def test_box_reader_folder(box_client_jwt_integration_testing: BoxClient):
    # Very slow test
    test_data = get_testing_data()
    if test_data["disable_folder_tests"]:
        raise pytest.skip(f"Slow integration tests are disabled.")

    reader = BoxReader(box_client=box_client_jwt_integration_testing)
    docs = reader.load_data(folder_id=test_data["test_folder_id"])
    assert len(docs) >= 1
