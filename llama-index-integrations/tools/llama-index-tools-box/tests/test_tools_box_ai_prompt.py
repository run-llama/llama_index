from box_sdk_gen import BoxClient

from llama_index.tools.box import BoxAIPromptToolSpec

from tests.conftest import get_testing_data


def test_box_tool_ai_prompt(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()

    box_tool = BoxAIPromptToolSpec(box_client=box_client_ccg_integration_testing)

    doc = box_tool.ai_prompt(
        file_id=test_data["test_ppt_id"], ai_prompt="Summarize the document"
    )

    assert doc.text is not None
