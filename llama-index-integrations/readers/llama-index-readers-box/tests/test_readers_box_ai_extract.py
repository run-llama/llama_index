import pytest
import json
from llama_index.core.readers.base import BaseReader
from llama_index.readers.box import BoxReaderAIExtract

from box_sdk_gen import BoxClient
from tests.conftest import get_testing_data


def test_class_name():
    names_of_base_classes = [b.__name__ for b in BoxReaderAIExtract.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_reader_init(box_client_ccg_unit_testing: BoxClient):
    reader = BoxReaderAIExtract(box_client=box_client_ccg_unit_testing)


####################################################################################################
# Integration tests
####################################################################################################

AI_PROMPT = (
    '{"doc_type","date","total","vendor","invoice_number","purchase_order_number"}'
)


def test_box_reader_ai_extract_whoami(box_client_ccg_integration_testing: BoxClient):
    me = box_client_ccg_integration_testing.users.get_user_me()
    assert me is not None


def test_box_reader_ai_extract_single_doc(
    box_client_ccg_integration_testing: BoxClient,
):
    reader = BoxReaderAIExtract(box_client=box_client_ccg_integration_testing)
    data = get_testing_data()
    docs = reader.load_data(file_ids=[data["test_txt_invoice_id"]], ai_prompt=AI_PROMPT)
    assert len(docs) == 1
    doc_0 = json.loads(docs[0].text)
    assert doc_0["doc_type"] == "Invoice"
    assert doc_0["total"] == "$1,050"
    assert doc_0["vendor"] == "Quasar Innovations"
    assert doc_0["invoice_number"] == "Q2468"
    assert doc_0["purchase_order_number"] == "003"


def test_box_reader_ai_extract_multi_doc(box_client_ccg_integration_testing: BoxClient):
    reader = BoxReaderAIExtract(box_client=box_client_ccg_integration_testing)
    data = get_testing_data()
    docs = reader.load_data(
        file_ids=[data["test_txt_invoice_id"], data["test_txt_po_id"]],
        ai_prompt=AI_PROMPT,
    )
    assert len(docs) == 2


def test_box_reader_ai_extract_folder(
    box_client_ccg_integration_testing: BoxClient,
):
    reader = BoxReaderAIExtract(box_client=box_client_ccg_integration_testing)
    data = get_testing_data()
    if data["disable_folder_tests"]:
        raise pytest.skip(f"Slow folder integration tests are disabled.")
    docs = reader.load_data(
        folder_id=data["test_folder_invoice_po_id"],
        ai_prompt=AI_PROMPT,
    )
    assert len(docs) > 2
