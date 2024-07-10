import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.box import BoxReaderTextExtraction

from box_sdk_gen import BoxClient
from tests.conftest import get_testing_data


def test_class_name():
    names_of_base_classes = [b.__name__ for b in BoxReaderTextExtraction.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_reader_init(box_client_ccg_unit_testing: BoxClient):
    reader = BoxReaderTextExtraction(box_client=box_client_ccg_unit_testing)


####################################################################################################
# Integration tests
####################################################################################################


def test_box_reader_text_extraction_whoami(
    box_client_ccg_integration_testing: BoxClient,
):
    me = box_client_ccg_integration_testing.users.get_user_me()
    assert me is not None


def test_box_reader_text_extraction_single_doc(
    box_client_ccg_integration_testing: BoxClient,
):
    reader = BoxReaderTextExtraction(box_client=box_client_ccg_integration_testing)
    data = get_testing_data()
    docs = reader.load_data(file_ids=[data["test_doc_id"]])
    assert len(docs) == 1


def test_box_reader_text_extraction_multi_doc(
    box_client_ccg_integration_testing: BoxClient,
):
    reader = BoxReaderTextExtraction(box_client=box_client_ccg_integration_testing)
    data = get_testing_data()
    docs = reader.load_data(
        file_ids=[data["test_doc_id"], data["test_txt_waiver_id"]],
    )
    assert len(docs) == 2


def test_box_reader_text_extraction_folder(
    box_client_ccg_integration_testing: BoxClient,
):
    reader = BoxReaderTextExtraction(box_client=box_client_ccg_integration_testing)
    data = get_testing_data()
    if data["disable_folder_tests"]:
        raise pytest.skip(f"Slow folder integration tests are disabled.")
    docs = reader.load_data(
        folder_id=data["test_folder_id"],
    )
    assert len(docs) > 2
