import pytest
import respx
from llama_index.embeddings.nvidia import NVIDIAEmbedding
import re
import httpx
import json


@pytest.fixture()
def mocked_route() -> respx.Route:
    all_urls = re.compile(r".*/embeddings")
    fake_response = httpx.Response(
        200, json={"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]}
    )
    with respx.mock:
        yield respx.post(all_urls).mock(return_value=fake_response)


@pytest.mark.parametrize("method_name", ["get_query_embedding", "get_text_embedding"])
@pytest.mark.parametrize("truncate", ["END", "START", "NONE"])
def test_single_tuncate(mocked_route: respx.Route, method_name: str, truncate: str):
    # call the method_name method
    getattr(NVIDIAEmbedding(api_key="BOGUS", truncate=truncate), method_name)("nvidia")

    assert mocked_route.called
    request = mocked_route.calls.last.request
    request_body = json.loads(request.read())
    assert "truncate" in request_body
    assert request_body["truncate"] == truncate


@pytest.mark.parametrize("method_name", ["aget_query_embedding", "aget_text_embedding"])
@pytest.mark.parametrize("truncate", ["END", "START", "NONE"])
@pytest.mark.asyncio()
async def test_asingle_tuncate(
    mocked_route: respx.Route, method_name: str, truncate: str
):
    # call the method_name method
    await getattr(NVIDIAEmbedding(api_key="BOGUS", truncate=truncate), method_name)(
        "nvidia"
    )

    assert mocked_route.called
    request = mocked_route.calls.last.request
    request_body = json.loads(request.read())
    assert "truncate" in request_body
    assert request_body["truncate"] == truncate


@pytest.mark.parametrize("method_name", ["get_text_embedding_batch"])
@pytest.mark.parametrize("truncate", ["END", "START", "NONE"])
def test_batch_tuncate(mocked_route: respx.Route, method_name: str, truncate: str):
    # call the method_name method
    getattr(NVIDIAEmbedding(api_key="BOGUS", truncate=truncate), method_name)(
        ["nvidia"]
    )

    assert mocked_route.called
    request = mocked_route.calls.last.request
    request_body = json.loads(request.read())
    assert "truncate" in request_body
    assert request_body["truncate"] == truncate


@pytest.mark.parametrize("method_name", ["aget_text_embedding_batch"])
@pytest.mark.parametrize("truncate", ["END", "START", "NONE"])
@pytest.mark.asyncio()
async def test_abatch_tuncate(
    mocked_route: respx.Route, method_name: str, truncate: str
):
    # call the method_name method
    await getattr(NVIDIAEmbedding(api_key="BOGUS", truncate=truncate), method_name)(
        ["nvidia"]
    )

    assert mocked_route.called
    request = mocked_route.calls.last.request
    request_body = json.loads(request.read())
    assert "truncate" in request_body
    assert request_body["truncate"] == truncate
