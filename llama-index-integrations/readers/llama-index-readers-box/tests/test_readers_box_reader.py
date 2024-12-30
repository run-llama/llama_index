import datetime
import pytest
from pathlib import Path
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


def test_box_reader_metadata(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()
    reader = BoxReader(box_client=box_client_ccg_integration_testing)
    docs = reader.load_data(file_ids=[test_data["test_csv_id"]])
    assert len(docs) == 1
    doc = docs[0]
    # check if metadata dictionary does not contain any datetime objects
    for v in doc.metadata.values():
        assert not isinstance(v, (datetime.datetime, datetime.date, datetime.time))


def test_box_reader_folder(box_client_ccg_integration_testing):
    # Very slow test
    test_data = get_testing_data()
    if test_data["disable_folder_tests"]:
        raise pytest.skip(f"Slow folder integration tests are disabled.")
    reader = BoxReader(box_client=box_client_ccg_integration_testing)

    docs = reader.load_data(folder_id=test_data["test_folder_id"])
    assert len(docs) >= 1


def test_box_reader_list_resources(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()
    reader = BoxReader(box_client=box_client_ccg_integration_testing)
    resource_id = test_data["test_csv_id"]
    resources = reader.list_resources(file_ids=[resource_id])
    assert len(resources) > 0
    assert resource_id in resources


def test_box_reader_get_resource_info(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()
    reader = BoxReader(box_client=box_client_ccg_integration_testing)
    resource_id = test_data["test_csv_id"]
    info = reader.get_resource_info(resource_id)
    assert info is not None
    assert info["id"] == resource_id


def test_box_reader_load_resource(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()
    reader = BoxReader(box_client=box_client_ccg_integration_testing)
    resource_id = test_data["test_csv_id"]
    doc = reader.load_resource(resource_id)
    assert doc is not None
    assert len(doc) == 1
    assert doc[0].metadata["box_file_id"] == resource_id


def test_box_reader_file_content(box_client_ccg_integration_testing):
    test_data = get_testing_data()
    reader = BoxReader(box_client=box_client_ccg_integration_testing)
    input_file: Path = Path(test_data["test_csv_id"])
    content = reader.read_file_content(input_file)
    assert content is not None
    assert len(content) > 0


def test_box_reader_search(box_client_ccg_integration_testing: BoxClient):
    reader = BoxReader(box_client=box_client_ccg_integration_testing)
    query = "invoice"
    resources = reader.search_resources(query=query)
    assert len(resources) > 0


def test_box_reader_search_by_metadata(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()
    reader = BoxReader(box_client=box_client_ccg_integration_testing)

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
