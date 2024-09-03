from pathlib import Path
import pytest
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


def test_box_reader_ai_prompt_folder(
    box_client_ccg_integration_testing: BoxClient,
):
    reader = BoxReaderAIPrompt(box_client=box_client_ccg_integration_testing)
    data = get_testing_data()
    if data["disable_folder_tests"]:
        raise pytest.skip(f"Slow folder integration tests are disabled.")
    docs = reader.load_data(
        folder_id=data["test_folder_id"],
        ai_prompt="summarize this document",
    )
    assert len(docs) > 2


def test_box_reader_list_resources(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()
    reader = BoxReaderAIPrompt(box_client=box_client_ccg_integration_testing)
    resource_id = test_data["test_csv_id"]
    resources = reader.list_resources(file_ids=[resource_id])
    assert len(resources) > 0
    assert resource_id in resources


def test_box_reader_get_resource_info(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()
    reader = BoxReaderAIPrompt(box_client=box_client_ccg_integration_testing)
    resource_id = test_data["test_csv_id"]
    info = reader.get_resource_info(resource_id)
    assert info is not None
    assert info["id"] == resource_id


def test_box_reader_load_resource(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()
    reader = BoxReaderAIPrompt(box_client=box_client_ccg_integration_testing)
    resource_id = test_data["test_txt_invoice_id"]
    docs = reader.load_resource(resource_id, ai_prompt="summarize this document")
    assert docs is not None
    assert len(docs) == 1
    assert docs[0].metadata["box_file_id"] == resource_id
    assert docs[0].text is not None


def test_box_reader_file_content(box_client_ccg_integration_testing):
    test_data = get_testing_data()
    reader = BoxReaderAIPrompt(box_client=box_client_ccg_integration_testing)
    input_file: Path = Path(test_data["test_csv_id"])
    content = reader.read_file_content(input_file)
    assert content is not None
    assert len(content) > 0


def test_box_reader_search(box_client_ccg_integration_testing: BoxClient):
    reader = BoxReaderAIPrompt(box_client=box_client_ccg_integration_testing)
    query = "invoice"
    resources = reader.search_resources(query=query)
    assert len(resources) > 0


def test_box_reader_search_by_metadata(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()
    reader = BoxReaderAIPrompt(box_client=box_client_ccg_integration_testing)

    # Parameters
    from_ = (
        test_data["metadata_enterprise_scope"]
        + "."
        + test_data["metadata_template_key"]
    )
    ancestor_folder_id = test_data["test_folder_invoice_po_id"]
    query = "documentType = :docType "
    query_params = {"docType": "Invoice"}

    resources = reader.search_resources_by_metadata(
        from_=from_,
        ancestor_folder_id=ancestor_folder_id,
        query=query,
        query_params=query_params,
    )
    assert len(resources) > 0
