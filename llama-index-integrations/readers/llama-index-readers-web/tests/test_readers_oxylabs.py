from unittest.mock import MagicMock, AsyncMock

import pytest
import sys

from llama_index.readers.web.oxylabs_web.base import OxylabsWebReader


READER_TEST_PARAM = pytest.param(
    [
        "https://sandbox.oxylabs.io/products/1",
        "https://sandbox.oxylabs.io/products/2",
    ],
    {
        "parse": True,
    },
    {
        "results": [{"content": {"key1": "value1", "key2": "value2"}}],
        "job": {"job_id": 42424242},
    },
    "# key1\n  value1\n\n# key2\n  value2\n",
    id="response_success",
)

skip_if_py39_or_lower = sys.version_info < (3, 10)


@pytest.mark.skipif(skip_if_py39_or_lower, reason="Pytest does not support Python 3.9")
@pytest.mark.parametrize(
    ("urls", "additional_params", "return_value", "expected_output"),
    [READER_TEST_PARAM],
)
def test_sync_oxylabs_reader(
    urls: list[str],
    additional_params: dict,
    return_value: dict,
    expected_output: str,
):
    reader = OxylabsWebReader(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
    )

    get_response_mock = MagicMock()
    get_response_mock.return_value = return_value
    reader.api.get_response = get_response_mock

    docs = reader.load_data(urls, additional_params)

    for doc in docs:
        assert doc.text == expected_output


@pytest.mark.skipif(skip_if_py39_or_lower, reason="Pytest does not support Python 3.9")
@pytest.mark.parametrize(
    ("urls", "additional_params", "return_value", "expected_output"),
    [READER_TEST_PARAM],
)
@pytest.mark.asyncio
async def test_async_oxylabs_reader(
    urls: list[str],
    additional_params: dict,
    return_value: dict,
    expected_output: str,
):
    reader = OxylabsWebReader(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
    )

    get_response_mock = AsyncMock()
    get_response_mock.return_value = return_value
    reader.async_api.get_response = get_response_mock

    docs = await reader.aload_data(urls, additional_params)

    for doc in docs:
        assert doc.text == expected_output


@pytest.mark.skipif(skip_if_py39_or_lower, reason="Pytest does not support Python 3.9")
def test_sync_oxylabs_reader_warns_on_dropped_url():
    """A url the API returns no result for should be reported, not dropped."""
    reader = OxylabsWebReader(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
    )

    ok = {
        "results": [{"content": {"key1": "value1"}}],
        "job": {"job_id": 42424242},
    }
    reader.api.get_response = MagicMock(side_effect=[ok, None])

    with pytest.warns(UserWarning, match="https://bad.example"):
        docs = reader.load_data(["https://good.example", "https://bad.example"])

    assert len(docs) == 1


@pytest.mark.skipif(skip_if_py39_or_lower, reason="Pytest does not support Python 3.9")
@pytest.mark.asyncio
async def test_async_oxylabs_reader_warns_on_dropped_url():
    reader = OxylabsWebReader(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
    )

    ok = {
        "results": [{"content": {"key1": "value1"}}],
        "job": {"job_id": 42424242},
    }
    reader.async_api.get_response = AsyncMock(side_effect=[ok, None])

    with pytest.warns(UserWarning, match="https://bad.example"):
        docs = await reader.aload_data(["https://good.example", "https://bad.example"])

    assert len(docs) == 1


@pytest.mark.skipif(skip_if_py39_or_lower, reason="Pytest does not support Python 3.9")
def test_reader_is_marked_remote():
    reader = OxylabsWebReader(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
    )
    assert reader.is_remote is True
