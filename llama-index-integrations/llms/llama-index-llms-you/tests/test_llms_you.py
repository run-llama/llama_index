from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.you import You


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in You.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_requests_set_timeout(monkeypatch):
    import types

    import requests
    from llama_index.llms.you import base

    calls = []

    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {}

    def fake_post(url, **kwargs):
        calls.append(kwargs)
        return _Response()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(
        base.sseclient,
        "SSEClient",
        lambda response: types.SimpleNamespace(events=lambda: []),
    )

    base._request(base.SMART_ENDPOINT, "key", query="hi")
    list(base._request_stream(base.SMART_ENDPOINT, "key", query="hi"))

    assert len(calls) == 2
    assert all(call.get("timeout") for call in calls)
