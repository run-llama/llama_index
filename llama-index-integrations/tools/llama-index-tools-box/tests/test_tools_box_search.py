from box_sdk_gen import BoxClient

from llama_index.tools.box import BoxSearchToolSpec, BoxSearchByMetadataToolSpec

from tests.conftest import get_testing_data


def test_box_too_search(box_client_ccg_integration_testing: BoxClient):
    query = "invoice"
    box_tool = BoxSearchToolSpec(box_client_ccg_integration_testing)
    docs = box_tool.search(query=query)
    assert len(docs) > 0


def test_box_tool_search_by_metadata(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()

    # Parameters
    from_ = (
        test_data["metadata_enterprise_scope"]
        + "."
        + test_data["metadata_template_key"]
    )
    ancestor_folder_id = test_data["test_folder_invoice_po_id"]
    query = "documentType = :docType "
    query_params = {"docType": "Invoice"}

    box_tool = BoxSearchByMetadataToolSpec(
        box_client=box_client_ccg_integration_testing
    )

    docs = box_tool.search(
        from_=from_,
        ancestor_folder_id=ancestor_folder_id,
        query=query,
        query_params=query_params,
    )
    assert len(docs) > 0
