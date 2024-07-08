from llama_index.core.readers.base import BaseReader
from llama_index.readers.box import BoxReaderAIPrompt

from box_sdk_gen import BoxClient
from tests.conftest import get_testing_data


def test_class_name():
    names_of_base_classes = [b.__name__ for b in BoxReaderAIPrompt.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_reader_init(box_client_ccg_unit_testing: BoxClient):
    reader = BoxReaderAIPrompt(box_client=box_client_ccg_unit_testing)


####################################################################################################
# Integration tests
####################################################################################################


def test_box_reader_ai_prompt_whoami(box_client_ccg_integration_testing: BoxClient):
    me = box_client_ccg_integration_testing.users.get_user_me()
    assert me is not None


def test_box_reader_ai_prompt_single_doc(box_client_ccg_integration_testing: BoxClient):
    reader = BoxReaderAIPrompt(box_client=box_client_ccg_integration_testing)
    data = get_testing_data()
    docs = reader.load_data(
        file_ids=[data["test_doc_id"]], ai_prompt="summarize this document"
    )
    assert len(docs) == 1


def test_box_reader_ai_prompt_multi_doc(box_client_ccg_integration_testing: BoxClient):
    reader = BoxReaderAIPrompt(box_client=box_client_ccg_integration_testing)
    data = get_testing_data()
    docs = reader.load_data(
        file_ids=[data["test_doc_id"], data["test_txt_waiver_id"]],
        ai_prompt="summarize this document",
    )
    assert len(docs) == 2


def test_box_reader_ai_prompt_multi_doc_group_prompt(
    box_client_ccg_integration_testing: BoxClient,
):
    reader = BoxReaderAIPrompt(box_client=box_client_ccg_integration_testing)
    data = get_testing_data()
    docs = reader.load_data(
        file_ids=[data["test_doc_id"], data["test_txt_waiver_id"]],
        ai_prompt="summarize this document",
        individual_document_prompt=False,
    )
    assert len(docs) == 2
