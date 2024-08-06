from box_sdk_gen import BoxClient

from llama_index.tools.box import BoxSearchToolSpec


def test_box_reader_search(box_client_ccg_integration_testing: BoxClient):
    query = "invoice"
    search_results = BoxSearchToolSpec()
    assert len(search_results) > 0
