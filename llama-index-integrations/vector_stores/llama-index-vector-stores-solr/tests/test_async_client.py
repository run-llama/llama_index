import asyncio
import sys
from typing import Any, Optional
from unittest import mock
from unittest.mock import MagicMock, patch

import aiohttp
import aiosolr
import pytest
from pydantic import ValidationError

from llama_index.vector_stores.solr.client import (
    AsyncSolrClient,
    SolrSelectResponse,
    SolrUpdateResponse,
)
from tests.conftest import compare_documents, params_delete_by_id, params_search_queries

_MODULE_PATH = "llama_index.vector_stores.solr.client.async_"


@patch(f"{_MODULE_PATH}.aiosolr.Client.update")
async def test_async_solr_client_add_valid(
    mock_aiosolr_add: MagicMock,
    mock_aiosolr_update_response: aiosolr.Response,
    mock_solr_update_response: SolrUpdateResponse,
    mock_solr_raw_input_documents: list[dict[str, Any]],
    mock_solr_updated_input_documents: list[dict[str, Any]],
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=mock_base_solr_url)
    mock_aiosolr_add.return_value = mock_aiosolr_update_response

    # WHEN
    actual_response = await async_client.add(mock_solr_raw_input_documents)

    # THEN
    mock_aiosolr_add.assert_called_once_with(data=mock_solr_updated_input_documents)
    assert actual_response == mock_solr_update_response


@patch(f"{_MODULE_PATH}.aiosolr.Client.update")
async def test_async_solr_client_add_aiosolr_error(
    mock_aiosolr_update: MagicMock,
    mock_solr_raw_input_documents: list[dict[str, Any]],
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=mock_base_solr_url)
    mock_aiosolr_update.side_effect = aiosolr.SolrError("some error")

    # WHEN / THEN
    with pytest.raises(
        ValueError,
        match=f"Error during Aiosolr call, type={aiosolr.SolrError}",
    ):
        await async_client.add(mock_solr_raw_input_documents)


@pytest.mark.uses_docker
async def test_async_solr_client_add_docker_solr(
    function_unique_solr_collection_url: str,
    mock_solr_raw_input_documents: list[dict[str, Any]],
    mock_solr_expected_retrieved_documents: list[dict[str, Any]],
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=function_unique_solr_collection_url)

    # WHEN
    await async_client.add(mock_solr_raw_input_documents)
    await asyncio.sleep(5)

    # THEN
    results = await async_client.search({"q": "*:*"})
    compare_documents(mock_solr_expected_retrieved_documents, results.response.docs)


@patch(f"{_MODULE_PATH}.aiosolr.Client.update")
async def test_async_solr_client_delete_by_query_valid(
    mock_aiosolr_update: MagicMock,
    mock_aiosolr_update_response: aiosolr.Response,
    mock_solr_update_response: SolrUpdateResponse,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=mock_base_solr_url)
    input_query_string = "id:doc1"
    expected_query = {"delete": {"query": "id:doc1"}}
    mock_aiosolr_update.return_value = mock_aiosolr_update_response

    # WHEN
    actual_response = await async_client.delete_by_query(input_query_string)

    # THEN
    mock_aiosolr_update.assert_called_once_with(data=expected_query)
    assert actual_response == mock_solr_update_response


@patch(f"{_MODULE_PATH}.aiosolr.Client.update")
async def test_async_solr_client_delete_by_query_aiosolr_error(
    mock_aiosolr_update: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=mock_base_solr_url)
    mock_aiosolr_update.side_effect = aiosolr.SolrError("some error")

    # WHEN / THEN
    with pytest.raises(
        ValueError,
        match=f"Error during Aiosolr call, type={aiosolr.SolrError}",
    ):
        await async_client.delete_by_query("id:doc1")


@pytest.mark.uses_docker
async def test_async_solr_client_delete_by_query_docker_solr(
    function_unique_solr_collection_url: str,
    mock_solr_raw_input_documents: list[dict[str, Any]],
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=function_unique_solr_collection_url)
    delete_query = "int_i:1"
    search_query = {"q": delete_query, "fl": "id,text_txt_en,score"}

    # WHEN
    # add, and ensure the docs are present
    await async_client.add(mock_solr_raw_input_documents)
    await asyncio.sleep(5)
    res_after_add = await async_client.search(search_query)
    assert len(res_after_add.response.docs) == 1

    # delete once we're sure they're there
    await async_client.delete_by_query(delete_query)
    await asyncio.sleep(5)

    # THEN
    res_after_del = await async_client.search(search_query)
    assert len(res_after_del.response.docs) == 0


@pytest.mark.parametrize(
    "input_ids", [["doc1"], ["doc1", "doc2"]], ids=["len(ids)==1", "len(ids)>1"]
)
@patch(f"{_MODULE_PATH}.aiosolr.Client.update")
async def test_async_solr_client_delete_by_id_valid(
    mock_aiosolr_update: MagicMock,
    mock_aiosolr_update_response: aiosolr.Response,
    mock_solr_update_response: SolrUpdateResponse,
    mock_base_solr_url: str,
    input_ids: list[str],
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=mock_base_solr_url)
    expected_query = {"delete": input_ids}
    mock_aiosolr_update.return_value = mock_aiosolr_update_response

    # WHEN
    actual_response = await async_client.delete_by_id(input_ids)

    # THEN
    mock_aiosolr_update.assert_called_once_with(data=expected_query)
    assert actual_response == mock_solr_update_response


@patch(f"{_MODULE_PATH}.aiosolr.Client.update")
async def test_async_solr_client_delete_by_id_empty_id_list(
    mock_aiohttp_client_session: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=mock_base_solr_url)

    # WHEN / THEN
    with pytest.raises(ValueError, match="The list of IDs to delete cannot be empty"):
        await async_client.delete_by_id([])


@patch(f"{_MODULE_PATH}.aiosolr.Client.update")
async def test_async_solr_client_delete_by_id_aiosolr_error(
    mock_aiosolr_update: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=mock_base_solr_url)
    mock_aiosolr_update.side_effect = aiosolr.SolrError("some error")

    # WHEN / THEN
    with pytest.raises(
        ValueError,
        match=f"Error during Aiosolr call, type={aiosolr.SolrError}",
    ):
        await async_client.delete_by_id(["doc1", "doc2"])


@params_delete_by_id
@pytest.mark.uses_docker
async def test_async_solr_client_delete_by_id_docker_solr(
    function_unique_solr_collection_url: str,
    mock_solr_raw_input_documents: list[dict[str, Any]],
    ids_to_delete: list[str],
    expected_remaining_ids: list[str],
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=function_unique_solr_collection_url)
    search_query = {"q": "*:*", "fl": "id"}
    expected_status = 200

    # WHEN
    # add, and ensure the docs are present
    await async_client.add(mock_solr_raw_input_documents)
    await asyncio.sleep(5)
    res_after_add = await async_client.search(search_query)
    assert len(res_after_add.response.docs) == len(mock_solr_raw_input_documents)

    actual_response = await async_client.delete_by_id(ids_to_delete)
    await asyncio.sleep(5)

    # THEN
    assert actual_response.response_header.status == expected_status
    res_after_del = await async_client.search(search_query)
    retrieved_ids = sorted([doc["id"] for doc in res_after_del.response.docs])
    assert retrieved_ids == expected_remaining_ids


@patch(f"{_MODULE_PATH}.aiosolr.Client.update")
async def test_async_solr_client_clear_collection_valid(
    mock_aiosolr_update: MagicMock,
    mock_aiosolr_update_response: aiosolr.Response,
    mock_solr_update_response: SolrUpdateResponse,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=mock_base_solr_url)
    expected_query = {"delete": {"query": "*:*"}}
    mock_aiosolr_update.return_value = mock_aiosolr_update_response

    # WHEN
    actual_response = await async_client.clear_collection()

    # THEN
    mock_aiosolr_update.assert_called_once_with(data=expected_query)
    assert actual_response == mock_solr_update_response


@patch(f"{_MODULE_PATH}.aiosolr.Client.update")
async def test_async_solr_client_clear_collection_aiosolr_error(
    mock_aiosolr_update: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=mock_base_solr_url)
    mock_aiosolr_update.side_effect = aiosolr.SolrError("some error")

    # WHEN / THEN
    with pytest.raises(
        ValueError,
        match=f"Error during Aiosolr call, type={aiosolr.SolrError}",
    ):
        await async_client.clear_collection()


@pytest.mark.uses_docker
async def test_async_solr_client_clear_collection_docker_solr(
    function_unique_solr_collection_url: str,
    mock_solr_raw_input_documents: list[dict[str, Any]],
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=function_unique_solr_collection_url)
    search_query = {"q": "*:*", "fl": "id,text_txt_en,score"}

    # WHEN
    # add, and ensure the docs are present
    await async_client.add(mock_solr_raw_input_documents)
    await asyncio.sleep(5)
    res_after_add = await async_client.search(search_query)
    assert len(res_after_add.response.docs) == len(mock_solr_raw_input_documents)

    # delete once we're sure they're there
    await async_client.clear_collection()
    await asyncio.sleep(5)

    # THEN
    res_after_del = await async_client.search(search_query)
    assert len(res_after_del.response.docs) == 0


@patch(f"{_MODULE_PATH}.aiosolr.Client.query")
async def test_async_solr_client_search_valid(
    mock_aiosolr_search: MagicMock,
    mock_aiosolr_search_results: aiosolr.Response,
    mock_solr_select_response: SolrSelectResponse,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=mock_base_solr_url)
    mock_aiosolr_search.return_value = mock_aiosolr_search_results

    # WHEN
    actual_response = await async_client.search({"q": "president", "fl": "*,score"})

    # THEN
    mock_aiosolr_search.assert_called_once_with(q="president", fl="*,score")
    assert actual_response == mock_solr_select_response


@patch(f"{_MODULE_PATH}.aiosolr.Client.query")
async def test_async_solr_client_search_aiosolr_error(
    mock_aiosolr_search: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=mock_base_solr_url)
    mock_aiosolr_search.side_effect = aiosolr.SolrError("some error")

    # WHEN / THEN
    with pytest.raises(
        ValueError,
        match=f"Error during Aiosolr call, type={aiosolr.SolrError}",
    ):
        await async_client.search({"q": "president", "fl": "*,score"})


@patch(f"{_MODULE_PATH}.aiosolr.Client.query")
async def test_async_solr_client_search_validation_error(
    mock_aiosolr_search: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=mock_base_solr_url)
    mock_aiosolr_search.side_effect = ValidationError("fake", [])

    # WHEN / THEN
    with pytest.raises(ValueError, match="Unexpected response format from Solr"):
        await async_client.search({"q": "president", "fl": "*,score"})


@params_search_queries
@pytest.mark.uses_docker
async def test_async_solr_client_search_docker_solr(
    function_unique_solr_collection_url: str,
    mock_solr_raw_input_documents: list[dict[str, Any]],
    mock_solr_expected_retrieved_documents: list[dict[str, Any]],
    input_query: dict[str, Any],
    expected_doc_indexes: list[int],
    requires_score: bool,
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=function_unique_solr_collection_url)
    expected_docs = [
        doc
        for i, doc in enumerate(mock_solr_expected_retrieved_documents)
        if i in expected_doc_indexes
    ]
    if requires_score:
        for doc in expected_docs:
            doc["score"] = mock.ANY

    # WHEN
    await async_client.add(mock_solr_raw_input_documents)
    await asyncio.sleep(5)
    results = await async_client.search(input_query)

    # THEN
    compare_documents(expected_docs, results.response.docs)


def test_async_solr_client_str_output(
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    async_client = AsyncSolrClient(base_url=mock_base_solr_url)

    # WHEN / THEN
    assert str(async_client) == f"AsyncSolrClient(base_url='{mock_base_solr_url}')"


@pytest.mark.parametrize(
    ("input_url", "expected_args"),
    [
        (
            "http://localhost:80/solr/my-collection",
            {
                "host": "localhost",
                "port": 80,
                "scheme": "http",
                "collection": "my-collection",
                "read_timeout": 10,
                "write_timeout": 10,
            },
        ),
        (
            "http://localhost:80/solr/my-collection/",
            {
                "host": "localhost",
                "port": 80,
                "scheme": "http",
                "collection": "my-collection",
                "read_timeout": 10,
                "write_timeout": 10,
            },
        ),
        (
            "http://0.0.0.0:80/solr/my-collection",
            {
                "host": "0.0.0.0",
                "port": 80,
                "scheme": "http",
                "collection": "my-collection",
                "read_timeout": 10,
                "write_timeout": 10,
            },
        ),
        (
            "http://0.0.0.0:80/solr/my-collection/",
            {
                "host": "0.0.0.0",
                "port": 80,
                "scheme": "http",
                "collection": "my-collection",
                "read_timeout": 10,
                "write_timeout": 10,
            },
        ),
        (
            "https://some.solr.host.com/api/solr/my-collection",
            {
                "connection_url": "https://some.solr.host.com/api/solr/my-collection",
                "read_timeout": 10,
                "write_timeout": 10,
            },
        ),
        (
            "https://some.solr.host.com/api/solr/my-collection/",
            {
                "connection_url": "https://some.solr.host.com/api/solr/my-collection",
                "read_timeout": 10,
                "write_timeout": 10,
            },
        ),
    ],
    ids=[
        "localhost URL",
        "localhost URL with trailing slash",
        "0.0.0.0 URL",
        "0.0.0.0 URL with trailing slash",
        "External URL",
        "External URL with trailing slash",
    ],
)
@pytest.mark.parametrize(
    ("input_headers", "expected_headers"),
    [
        (None, {}),
        ({}, {}),
        (
            {"Content-Type": "application/json"},
            {"Content-Type": "application/json"},
        ),
    ],
    ids=["null value", "empty dict", "valid header dict"],
)
@pytest.mark.parametrize(
    "client_kwargs",
    [{}, {"ttl_dns_cache": 4800}],
    ids=["empty dict", "valid extra kwargs"],
)
@patch(f"{_MODULE_PATH}.aiosolr.Client", autospec=True)
async def test_async_solr_client_build_client(
    mock_aiosolr_client_init: MagicMock,
    input_url: str,
    expected_args: dict[str, Any],
    input_headers: Optional[dict[str, str]],
    expected_headers: dict[str, str],
    client_kwargs: dict[str, Any],
) -> None:
    # GIVEN
    mock_aiosolr_client_instance = mock_aiosolr_client_init.return_value
    mock_session = MagicMock(spec=aiohttp.ClientSession, headers={})
    mock_aiosolr_client_instance.session = mock_session
    expected_args = {**expected_args, **client_kwargs}

    # WHEN
    client = AsyncSolrClient(
        base_url=input_url,
        request_timeout_sec=10,
        headers=input_headers,
        **client_kwargs,
    )
    # ensure the inner client gets built
    _ = await client._build_client()

    # THEN
    # handle py3.9
    if sys.version_info < (3, 10):
        expected_args["timeout"] = expected_args["read_timeout"]
        del expected_args["read_timeout"]
        del expected_args["write_timeout"]

    mock_aiosolr_client_init.assert_called_once_with(**expected_args)
    assert mock_aiosolr_client_instance.session.headers == expected_headers


@pytest.mark.parametrize(
    ("input_url", "input_timeout"),
    [("https://some.solr.host", -1), (" ", 10), ("", 10), ("", -1)],
    ids=[
        "Negative timeout value",
        "Non-empty whitespace URL",
        "Empty URL",
        "Empty URL + negative timeout",
    ],
)
def test_async_solr_client_build_client_invalid_params(
    input_url: str, input_timeout: int
) -> None:
    # WHEN / THEN
    with pytest.raises(ValueError):
        _ = AsyncSolrClient(base_url=input_url, request_timeout_sec=input_timeout)
