from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.widespread import WidespreadToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in WidespreadToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions_read_only():
    assert WidespreadToolSpec.spec_functions == [
        "list_bounties",
        "list_markets",
        "feed_search",
        "get_feed_item",
        "list_runners",
        "get_state",
    ]


def test_defaults():
    spec = WidespreadToolSpec()
    assert spec.base_url == "https://widespread-bridge.widespread.workers.dev"
    assert spec.timeout_seconds == 30
