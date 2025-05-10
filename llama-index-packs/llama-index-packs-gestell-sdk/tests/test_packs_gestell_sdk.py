import pytest
from llama_index.packs.gestell_sdk import GestellSDKPack
from llama_index.core.llama_pack import BaseLlamaPack


def test_inheritance():
    assert issubclass(GestellSDKPack, BaseLlamaPack)


@pytest.mark.asyncio
async def test_search(monkeypatch):
    class MockResponse:
        result = [{"content": "hello"}]

    async def mock_search(*args, **kwargs):
        return MockResponse()

    pack = GestellSDKPack(collection_id="test", api_key="test")

    helper_cls = type(pack._gestell.query)
    monkeypatch.setattr(helper_cls, "search", mock_search)

    out = await pack.asearch("hello")
    assert out[0]["content"] == "hello"
