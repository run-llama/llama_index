from unittest.mock import Mock

import pytest

from llama_index.readers.notion import NotionPageReader


def block(block_id, text, has_children=False):
    return {
        "id": block_id,
        "type": "paragraph",
        "paragraph": {"rich_text": [{"text": {"content": text}}]},
        "has_children": has_children,
    }


@pytest.mark.parametrize("paginated", [False, True])
def test_read_page_keeps_block_id_and_sends_cursor(monkeypatch, paginated):
    calls = []
    pages = {
        ("page", None): {
            "results": [block("child", "Parent", has_children=True)],
            "next_cursor": "page-cursor" if paginated else None,
        },
        ("child", None): {
            "results": [block("leaf-a", "Child A")],
            "next_cursor": "child-cursor" if paginated else None,
        },
        ("child", "child-cursor"): {
            "results": [block("leaf-b", "Child B")],
            "next_cursor": None,
        },
        ("page", "page-cursor"): {
            "results": [block("sibling", "Sibling")],
            "next_cursor": None,
        },
    }

    def request(method, url, **kwargs):
        assert method == "GET"
        assert not kwargs.get("json")
        block_id = url.split("/")[-2]
        cursor = (kwargs.get("params") or {}).get("start_cursor")
        calls.append((block_id, cursor))
        response = Mock()
        response.json.return_value = pages[(block_id, cursor)]
        return response

    monkeypatch.setattr("requests.request", request)
    reader = NotionPageReader(integration_token="placeholder-token")
    expected = "Parent\n\tChild A"
    expected_calls = [("page", None), ("child", None)]
    if paginated:
        expected += "\n\tChild B\nSibling"
        expected_calls += [("child", "child-cursor"), ("page", "page-cursor")]

    assert reader.read_page("page") == expected
    assert calls == expected_calls
    calls.clear()
    assert reader.read_page("page") == expected
    assert calls == expected_calls
