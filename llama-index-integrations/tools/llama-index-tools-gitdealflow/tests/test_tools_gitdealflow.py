from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.gitdealflow import GitDealFlowToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in GitDealFlowToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions_registered():
    tool_spec = GitDealFlowToolSpec()
    assert "get_signals_summary" in tool_spec.spec_functions
    assert "get_startup_signal" in tool_spec.spec_functions
    assert "search_startups_by_sector" in tool_spec.spec_functions
    assert "answer_question" in tool_spec.spec_functions
    assert "get_methodology" in tool_spec.spec_functions


def test_default_base_url():
    tool_spec = GitDealFlowToolSpec()
    assert tool_spec.base_url == "https://signals.gitdealflow.com"


def test_custom_base_url_strips_trailing_slash():
    tool_spec = GitDealFlowToolSpec(base_url="https://example.com/")
    assert tool_spec.base_url == "https://example.com"


def test_to_tool_list_has_all_functions():
    tool_spec = GitDealFlowToolSpec()
    tools = tool_spec.to_tool_list()
    names = {t.metadata.name for t in tools}
    assert names == {
        "get_signals_summary",
        "get_startup_signal",
        "search_startups_by_sector",
        "answer_question",
        "get_methodology",
    }
