from box_sdk_gen import BoxClient

from llama_index.tools.box import BoxAIExtractToolSpec

from tests.conftest import get_testing_data

AI_PROMPT = (
    '{"doc_type","date","total","vendor","invoice_number","purchase_order_number"}'
)


def test_box_tool_ai_extract(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()

    box_tool = BoxAIExtractToolSpec(box_client=box_client_ccg_integration_testing)

    doc = box_tool.ai_extract(
        file_id=test_data["test_txt_invoice_id"], ai_prompt=AI_PROMPT
    )

    assert doc.text is not None
