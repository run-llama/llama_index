import os
import json
import pytest
from llama_index.tools.bocha_search import BochaSearchToolSpec

BOCHA_API_KEY = os.environ.get("BOCHA_API_KEY")
SKIP_REASON = (
    "Set BOCHA_API_KEY environment variable to run integration tests with real API"
)


@pytest.mark.skipif(not BOCHA_API_KEY, reason=SKIP_REASON)
def test_real_bocha_search() -> None:
    """Test bocha_search with the real API key."""
    spec = BochaSearchToolSpec(api_key=BOCHA_API_KEY)
    docs = spec.bocha_search(query="温州医科大学", count=1)

    assert len(docs) == 1
    response_json = json.loads(docs[0].text)
    assert response_json.get("code") == 200
    assert "data" in response_json


@pytest.mark.skipif(not BOCHA_API_KEY, reason=SKIP_REASON)
@pytest.mark.asyncio
async def test_real_abocha_search() -> None:
    """Test abocha_search asynchronously with the real API key."""
    spec = BochaSearchToolSpec(api_key=BOCHA_API_KEY)
    docs = await spec.abocha_search(query="温州医科大学", count=1)

    assert len(docs) == 1
    response_json = json.loads(docs[0].text)
    assert response_json.get("code") == 200
    assert "data" in response_json
