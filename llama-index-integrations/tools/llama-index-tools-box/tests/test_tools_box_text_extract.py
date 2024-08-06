from box_sdk_gen import BoxClient

from llama_index.tools.box import BoxTextExtractToolSpec

from tests.conftest import get_testing_data


def test_box_tool_extract(box_client_ccg_integration_testing: BoxClient):
    test_data = get_testing_data()

    box_tool = BoxTextExtractToolSpec(box_client=box_client_ccg_integration_testing)

    doc = box_tool.extract(test_data["test_ppt_id"])

    assert doc.text is not None
