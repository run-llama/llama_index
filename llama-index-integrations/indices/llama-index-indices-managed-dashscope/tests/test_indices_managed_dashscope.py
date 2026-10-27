from llama_index.indices.managed.dashscope import DashScopeCloudIndex
from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.core.schema import QueryBundle, TextNode
from llama_index.indices.managed.dashscope.retriever import DashScopeCloudRetriever


def test_class():
    names_of_base_classes = [b.__name__ for b in DashScopeCloudIndex.__mro__]
    assert BaseManagedIndex.__name__ in names_of_base_classes


def test_retrieve_sends_query_history_param(mocker):
    mocker.patch(
        "llama_index.indices.managed.dashscope.retriever.utils.get_pipeline_id",
        return_value="pipeline-id",
    )
    captured_params = {}

    def mock_post(base_url, headers, params):
        captured_params.update(params)
        node = TextNode(text="result")
        return {"nodes": [{"node": node.model_dump(), "score": 1.0}]}

    mocker.patch(
        "llama_index.indices.managed.dashscope.retriever.utils.post",
        side_effect=mock_post,
    )

    retriever = DashScopeCloudRetriever(
        index_name="test-index",
        api_key="test-api-key",
        workspace_id="test-workspace",
    )
    query_history = [{"role": "user", "content": "previous question"}]

    nodes = retriever._retrieve(
        QueryBundle("current question"), query_history=query_history
    )

    assert captured_params["query_history"] == query_history
    assert "query_hisory" not in captured_params
    assert len(nodes) == 1
