import json
from typing import Any, Dict, List

import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.dchub import DCHubToolSpec
from llama_index.tools.dchub.base import _parse_rpc

ANSWER = (
    "Ashburn: DCPI verdict (preview). Source: DC Hub (dchub.cloud), as of 2026-11-23"
)


class FakeResponse:
    def __init__(self, text: str, status: int = 200) -> None:
        self.text = text
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _ok(text: str = ANSWER) -> str:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": text}]},
        }
    )


@pytest.fixture()
def calls(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    seen: List[Dict[str, Any]] = []

    def fake_post(
        url: str, json: Dict[str, Any], headers: Dict[str, str], timeout: float
    ):
        seen.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse(_ok())

    monkeypatch.setattr("llama_index.tools.dchub.base.requests.post", fake_post)
    monkeypatch.delenv("DCHUB_API_KEY", raising=False)
    return seen


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in DCHubToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions_become_tools() -> None:
    tools = DCHubToolSpec().to_tool_list()
    names = [t.metadata.name for t in tools]
    assert names == [
        "execute_plan",
        "get_market_dcpi_rank",
        "get_interconnection_queue",
        "get_grid_intelligence",
        "search_facilities",
    ]
    assert "Use this first" in tools[0].metadata.description


def test_execute_plan_sends_tools_call_and_drops_unset_args(calls) -> None:
    out = DCHubToolSpec().execute_plan("DCPI for Ashburn?", market="ashburn")
    assert out == ANSWER
    body = calls[0]["json"]
    assert calls[0]["url"] == "https://dchub.cloud/mcp"
    assert body["method"] == "tools/call"
    assert body["params"] == {
        "name": "execute_plan",
        "arguments": {"intent": "DCPI for Ashburn?", "market": "ashburn"},
    }
    assert "X-API-Key" not in calls[0]["headers"]
    assert "text/event-stream" in calls[0]["headers"]["Accept"]


def test_api_key_from_env_goes_in_header(calls, monkeypatch) -> None:
    monkeypatch.setenv("DCHUB_API_KEY", "test-key")
    DCHubToolSpec().get_market_dcpi_rank("ashburn")
    assert calls[0]["headers"]["X-API-Key"] == "test-key"
    assert calls[0]["json"]["params"]["arguments"] == {"market_slug": "ashburn"}


def test_each_tool_maps_to_its_mcp_name(calls) -> None:
    spec = DCHubToolSpec(endpoint="https://example.test/mcp")
    spec.get_interconnection_queue(iso="PJM")
    spec.get_grid_intelligence(market="ashburn")
    spec.search_facilities(query="Ashburn")
    names = [c["json"]["params"]["name"] for c in calls]
    assert names == [
        "get_interconnection_queue",
        "get_grid_intelligence",
        "search_facilities",
    ]
    assert calls[2]["json"]["params"]["arguments"] == {"query": "Ashburn", "limit": 10}
    assert all(c["url"] == "https://example.test/mcp" for c in calls)


def test_sse_payload_is_parsed() -> None:
    sse = "event: message\ndata: " + _ok("hello") + "\n\n"
    assert _parse_rpc(sse)["result"]["content"][0]["text"] == "hello"


def test_rpc_error_is_returned_as_text(monkeypatch) -> None:
    err = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "bad"}}
    )
    monkeypatch.setattr(
        "llama_index.tools.dchub.base.requests.post",
        lambda *a, **k: FakeResponse(err),
    )
    assert DCHubToolSpec().execute_plan("x") == "DC Hub error -32602: bad"


def test_tool_error_flag(monkeypatch) -> None:
    payload = json.dumps(
        {
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": "timed out"}],
            }
        }
    )
    monkeypatch.setattr(
        "llama_index.tools.dchub.base.requests.post",
        lambda *a, **k: FakeResponse(payload),
    )
    assert DCHubToolSpec().execute_plan("x") == "DC Hub tool error: timed out"


def test_http_error_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        "llama_index.tools.dchub.base.requests.post",
        lambda *a, **k: FakeResponse("", status=503),
    )
    with pytest.raises(RuntimeError):
        DCHubToolSpec().execute_plan("x")


def test_garbage_payload_raises() -> None:
    with pytest.raises(ValueError):
        _parse_rpc("event: ping\n\n")
