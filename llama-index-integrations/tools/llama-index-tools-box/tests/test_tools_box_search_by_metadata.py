import pytest
import openai
from box_sdk_gen import BoxClient

from llama_index.tools.box import (
    BoxSearchByMetadataToolSpec,
    BoxSearchByMetadataOptions,
)
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
    query_params = '{"docType": "Invoice"}'

    # Search options
    options = BoxSearchByMetadataOptions(
        from_=from_,
        ancestor_folder_id=ancestor_folder_id,
        query=query,
    )

    box_tool = BoxSearchByMetadataToolSpec(
        box_client=box_client_ccg_integration_testing, options=options
    )

    docs = box_tool.search(
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

    # Parameters
    from_ = (
        test_data["metadata_enterprise_scope"]
        + "."
        + test_data["metadata_template_key"]
    )
    ancestor_folder_id = test_data["test_folder_invoice_po_id"]
    query = "documentType = :docType "
    query_params = '{"docType": "Invoice"}'

    # Search options
    options = BoxSearchByMetadataOptions(
        from_=from_,
        ancestor_folder_id=ancestor_folder_id,
        query=query,
    )

    box_tool = BoxSearchByMetadataToolSpec(
        box_client=box_client_ccg_integration_testing, options=options
    )

    openai.api_key = openai_api_key

    agent = OpenAIAgent.from_tools(
        box_tool.to_tool_list(),
        verbose=True,
    )

    answer = agent.chat(
        f"search all documents using the query_params as the key value pair of  {query_params} "
    )
    print(answer)
    assert answer is not None
