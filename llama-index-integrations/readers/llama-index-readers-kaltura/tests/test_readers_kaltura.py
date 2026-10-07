from llama_index.core.readers.base import BaseReader
from llama_index.readers.kaltura_esearch import KalturaESearchReader


def test_class():
    names_of_base_classes = [b.__name__ for b in KalturaESearchReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_get_json_transcript_sets_timeout(monkeypatch):
    import requests
    from unittest.mock import MagicMock

    calls = []

    class _Response:
        def json(self):
            return {"objects": []}

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        return _Response()

    monkeypatch.setattr(requests, "get", fake_get)

    reader = KalturaESearchReader.__new__(KalturaESearchReader)
    reader.client = MagicMock()

    assert reader._get_json_transcript("asset-id") == {"objects": []}
    assert calls[0].get("timeout")
