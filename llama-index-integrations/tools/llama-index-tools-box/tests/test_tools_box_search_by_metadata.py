import pytest
import openai
from box_sdk_gen import BoxClient

from llama_index.tools.box import BoxSearchByMetadataToolSpec
from llama_index.agent.openai import OpenAIAgent

from tests.conftest import get_testing_data


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


def test_box_tool_search_by_metadata_agent(
    box_client_ccg_integration_testing: BoxClient,
):
    test_data = get_testing_data()
    openai_api_key = test_data["openai_api_key"]

    if openai_api_key is None:
        raise pytest.skip("OpenAI API key is not provided.")

    box_tool_spec = BoxSearchByMetadataToolSpec(box_client_ccg_integration_testing)

    openai.api_key = openai_api_key

    agent = OpenAIAgent.from_tools(
        box_tool_spec.to_tool_list(),
        verbose=True,
    )

    answer = agent.chat("search all invoices")
    assert answer is not None
