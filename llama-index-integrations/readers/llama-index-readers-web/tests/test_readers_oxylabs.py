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
