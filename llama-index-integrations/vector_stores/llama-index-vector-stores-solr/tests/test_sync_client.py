import time
from typing import Any, Optional
from unittest import mock
from unittest.mock import MagicMock, patch

import pysolr
import pytest
import requests
from pydantic import ValidationError

from llama_index.vector_stores.solr.client import (
    SolrSelectResponse,
    SolrUpdateResponse,
    SyncSolrClient,
)
from tests.conftest import compare_documents, params_delete_by_id, params_search_queries

_MODULE_PATH = "llama_index.vector_stores.solr.client.sync"


@patch(f"{_MODULE_PATH}.pysolr.Solr.add")
def test_sync_solr_client_add_valid(
    mock_pysolr_add: MagicMock,
    mock_solr_raw_input_documents: list[dict[str, Any]],
    mock_solr_updated_input_documents: list[dict[str, Any]],
    mock_base_solr_url: str,
    mock_pysolr_update_response: str,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_add.return_value = mock_pysolr_update_response

    # WHEN
    sync_client.add(mock_solr_raw_input_documents)

    # THEN
    mock_pysolr_add.assert_called_once_with(mock_solr_updated_input_documents)


@patch(f"{_MODULE_PATH}.pysolr.Solr.add")
def test_sync_solr_client_add_pysolr_error(
    mock_pysolr_add: MagicMock,
    mock_solr_raw_input_documents: list[dict[str, Any]],
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_add.side_effect = pysolr.SolrError

    # WHEN / THEN
    with pytest.raises(
        ValueError, match=f"Error during Pysolr call, type={pysolr.SolrError}"
    ):
        sync_client.add(mock_solr_raw_input_documents)


@patch(f"{_MODULE_PATH}.pysolr.Solr.add")
def test_sync_solr_client_add_validation_error(
    mock_pysolr_add: MagicMock,
    mock_solr_raw_input_documents: list[dict[str, Any]],
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_add.return_value = '{"bad_response": "dict"}'

    # WHEN / THEN
    with pytest.raises(ValueError, match="Unexpected response format from Solr"):
        sync_client.add(mock_solr_raw_input_documents)


@pytest.mark.uses_docker
def test_sync_solr_client_add_docker_solr(
    function_unique_solr_collection_url: str,
    mock_solr_raw_input_documents: list[dict[str, Any]],
    mock_solr_expected_retrieved_documents: list[dict[str, Any]],
) -> None:
    # GIVEN
    client = SyncSolrClient(base_url=function_unique_solr_collection_url)
    query = {"q": "*:*", "fl": ",".join(mock_solr_raw_input_documents[0])}

    # WHEN
    client.add(mock_solr_raw_input_documents)
    time.sleep(5)

    # THEN
    results = client.search(query)
    compare_documents(mock_solr_expected_retrieved_documents, results.response.docs)


@patch(f"{_MODULE_PATH}.pysolr.Solr.delete")
def test_sync_solr_client_delete_by_query_valid(
    mock_pysolr_delete: MagicMock,
    mock_base_solr_url: str,
    mock_solr_delete_response_xml: str,
    mock_solr_delete_response: SolrUpdateResponse,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    input_query_string = "id:doc1"
    mock_pysolr_delete.return_value = mock_solr_delete_response_xml

    # WHEN
    actual_response = sync_client.delete_by_query(input_query_string)

    # THEN
    mock_pysolr_delete.assert_called_once_with(q=input_query_string, id=None)
    assert actual_response == mock_solr_delete_response


@patch(f"{_MODULE_PATH}.pysolr.Solr.delete")
def test_sync_solr_client_delete_by_query_pysolr_error(
    mock_pysolr_delete: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_delete.side_effect = pysolr.SolrError

    # WHEN / THEN
    with pytest.raises(
        ValueError, match=f"Error during Pysolr call, type={pysolr.SolrError}"
    ):
        sync_client.delete_by_query("id:doc1")


@patch(f"{_MODULE_PATH}.pysolr.Solr.delete")
def test_sync_solr_client_delete_by_query_validation_error(
    mock_pysolr_delete: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_delete.return_value = '{"bad_response": "dict"}'

    # WHEN / THEN
    with pytest.raises(ValueError):
        sync_client.delete_by_query("id:doc1")


@pytest.mark.uses_docker
def test_sync_solr_client_delete_by_query_docker_solr(
    function_unique_solr_collection_url: str,
    mock_solr_raw_input_documents: list[dict[str, Any]],
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=function_unique_solr_collection_url)
    delete_query = "int_i:1"
    search_query = {"q": delete_query, "fl": "id,text_txt_en,score"}

    # WHEN
    # add, and ensure the docs are present
    sync_client.add(mock_solr_raw_input_documents)
    time.sleep(5)

    res_after_add = sync_client.search(search_query)
    assert len(res_after_add.response.docs) == 1

    # delete once we're sure they're there
    sync_client.delete_by_query(delete_query)
    time.sleep(5)

    # THEN
    res_after_del = sync_client.search(search_query)
    assert len(res_after_del.response.docs) == 0


@pytest.mark.parametrize(
    "input_ids", [["doc1"], ["doc1", "doc2"]], ids=["len(ids)==1", "len(ids)>1"]
)
@patch(f"{_MODULE_PATH}.pysolr.Solr.delete")
def test_sync_solr_client_delete_by_id_valid(
    mock_pysolr_delete: MagicMock,
    mock_base_solr_url: str,
    input_ids: list[str],
    mock_solr_delete_response_xml: str,
    mock_solr_delete_response: SolrUpdateResponse,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_delete.return_value = mock_solr_delete_response_xml

    # WHEN
    actual_response = sync_client.delete_by_id(input_ids)

    # THEN
    mock_pysolr_delete.assert_called_once_with(id=input_ids, q=None)
    assert actual_response == mock_solr_delete_response


@patch(f"{_MODULE_PATH}.pysolr.Solr.delete")
def test_sync_solr_client_delete_by_id_pysolr_error(
    mock_pysolr_delete: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_delete.side_effect = pysolr.SolrError

    # WHEN / THEN
    with pytest.raises(
        ValueError, match=f"Error during Pysolr call, type={pysolr.SolrError}"
    ):
        sync_client.delete_by_id(["doc1", "doc2"])


@patch(f"{_MODULE_PATH}.pysolr.Solr.delete")
def test_sync_solr_client_delete_by_id_validation_error(
    mock_pysolr_delete: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_delete.return_value = '{"bad_response": "dict"}'

    # WHEN / THEN
    with pytest.raises(ValueError):
        sync_client.delete_by_id(["doc1", "doc2"])


@params_delete_by_id
@pytest.mark.uses_docker
def test_sync_solr_client_delete_by_id_docker_solr(
    function_unique_solr_collection_url: str,
    mock_solr_raw_input_documents: list[dict[str, Any]],
    ids_to_delete: list[str],
    expected_remaining_ids: list[str],
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=function_unique_solr_collection_url)
    search_query = {"q": "*:*", "fl": "id"}

    # WHEN
    # add, and ensure the docs are present
    sync_client.add(mock_solr_raw_input_documents)
    time.sleep(5)
    res_after_add = sync_client.search(search_query)
    assert len(res_after_add.response.docs) == len(mock_solr_raw_input_documents)

    # delete once we're sure they're there
    actual_response = sync_client.delete_by_id(ids_to_delete)
    time.sleep(5)

    # THEN
    assert actual_response.response_header.status == 0
    res_after_del = sync_client.search(search_query)
    retrieved_ids = sorted([doc["id"] for doc in res_after_del.response.docs])
    assert retrieved_ids == expected_remaining_ids


@patch(f"{_MODULE_PATH}.pysolr.Solr.delete")
def test_sync_solr_client_clear_collection_valid(
    mock_pysolr_delete: MagicMock,
    mock_base_solr_url: str,
    mock_solr_delete_response_xml: str,
    mock_solr_delete_response: SolrUpdateResponse,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_delete.return_value = mock_solr_delete_response_xml

    # WHEN
    actual_response = sync_client.clear_collection()

    # THEN
    mock_pysolr_delete.assert_called_once_with(id=None, q="*:*")
    assert actual_response == mock_solr_delete_response


@patch(f"{_MODULE_PATH}.pysolr.Solr.delete")
def test_sync_solr_client_clear_collection_pysolr_error(
    mock_pysolr_delete: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_delete.side_effect = pysolr.SolrError

    # WHEN / THEN
    with pytest.raises(
        ValueError, match=f"Error during Pysolr call, type={pysolr.SolrError}"
    ):
        sync_client.clear_collection()


@patch(f"{_MODULE_PATH}.pysolr.Solr.delete")
def test_sync_solr_client_clear_collection_validation_error(
    mock_pysolr_delete: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_delete.return_value = '{"bad_response": "dict"}'

    # WHEN / THEN
    with pytest.raises(ValueError):
        sync_client.clear_collection()


@pytest.mark.uses_docker
def test_sync_solr_client_clear_collection_docker_solr(
    function_unique_solr_collection_url: str,
    mock_solr_raw_input_documents: list[dict[str, Any]],
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=function_unique_solr_collection_url)
    search_query = {"q": "*:*", "fl": "id,text_txt_en,score"}

    # WHEN
    # add, and ensure the docs are present
    sync_client.add(mock_solr_raw_input_documents)
    time.sleep(5)
    res_after_add = sync_client.search(search_query)
    assert len(res_after_add.response.docs) == len(mock_solr_raw_input_documents)

    # delete once we're sure they're there
    sync_client.clear_collection()
    time.sleep(5)

    # THEN
    res_after_del = sync_client.search(search_query)
    assert len(res_after_del.response.docs) == 0


@patch(f"{_MODULE_PATH}.pysolr.Solr.search")
def test_sync_solr_client_search_valid(
    mock_pysolr_search: MagicMock,
    mock_pysolr_search_results: pysolr.Results,
    mock_solr_select_response: SolrSelectResponse,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_search.return_value = mock_pysolr_search_results

    # WHEN
    actual_response = sync_client.search({"q": "president", "fl": "*,score"})

    # THEN
    mock_pysolr_search.assert_called_once_with(q="president", fl="*,score")
    assert actual_response == mock_solr_select_response


@patch(f"{_MODULE_PATH}.pysolr.Solr.search")
def test_sync_solr_client_search_pysolr_error(
    mock_pysolr_search: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_search.side_effect = pysolr.SolrError

    # WHEN / THEN
    with pytest.raises(
        ValueError, match=f"Error during Pysolr call, type={pysolr.SolrError}"
    ):
        sync_client.search({"q": "president", "fl": "*,score"})


@patch(f"{_MODULE_PATH}.pysolr.Solr.search")
def test_sync_solr_client_search_validation_error(
    mock_pysolr_search: MagicMock,
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)
    mock_pysolr_search.side_effect = ValidationError("fake", [])

    # WHEN / THEN
    with pytest.raises(ValueError, match="Unexpected response format from Solr"):
        sync_client.search({"q": "president", "fl": "*,score"})


@params_search_queries
@pytest.mark.uses_docker
def test_sync_solr_client_search_docker_solr(
    function_unique_solr_collection_url: str,
    mock_solr_raw_input_documents: list[dict[str, Any]],
    mock_solr_expected_retrieved_documents: list[dict[str, Any]],
    input_query: dict[str, Any],
    expected_doc_indexes: list[int],
    requires_score: bool,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=function_unique_solr_collection_url)
    expected_docs = [
        doc
        for i, doc in enumerate(mock_solr_expected_retrieved_documents)
        if i in expected_doc_indexes
    ]
    if requires_score:
        for doc in expected_docs:
            doc["score"] = mock.ANY

    # WHEN
    sync_client.add(mock_solr_raw_input_documents)
    time.sleep(5)
    actual_results = sync_client.search(input_query)

    # THEN
    compare_documents(expected_docs, actual_results.response.docs)


def test_sync_solr_client_str_output(
    mock_base_solr_url: str,
) -> None:
    # GIVEN
    sync_client = SyncSolrClient(base_url=mock_base_solr_url)

    # WHEN / THEN
    assert str(sync_client) == f"SyncSolrClient(base_url='{mock_base_solr_url}')"


@pytest.mark.parametrize(
    "input_url",
    [
        "http://localhost:80/solr/my-collection",
        "http://0.0.0.0:80/solr/my-collection",
        "https://some.solr.host.com/api/solr/my-collection",
    ],
    ids=["localhost URL", "0.0.0.0 URL", "External URL"],
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
    [{}, {"search_handler": "search2"}],
    ids=["empty dict", "valid extra kwargs"],
)
@patch(f"{_MODULE_PATH}.pysolr.Solr", autospec=True)
def test_sync_solr_client_build_client(
    mock_pysolr_solr_init: MagicMock,
    input_url: str,
    input_headers: Optional[dict[str, str]],
    expected_headers: dict[str, str],
    client_kwargs: dict[str, str],
) -> None:
    # GIVEN
    mock_pysolr_solr_instance = mock_pysolr_solr_init.return_value
    mock_session = MagicMock(spec=requests.Session, headers={})
    mock_pysolr_solr_instance.get_session.return_value = mock_session
    expected_args = {"url": input_url, "timeout": 10, **client_kwargs}

    # WHEN
    client = SyncSolrClient(
        base_url=input_url,
        request_timeout_sec=10,
        headers=input_headers,
        **client_kwargs,
    )
    # ensure the inner client gets built
    _ = client._build_client()

    # THEN
    mock_pysolr_solr_init.assert_called_once_with(**expected_args)
    assert mock_session.headers == expected_headers


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
def test_sync_solr_client_build_client_invalid_params(
    input_url: str, input_timeout: int
) -> None:
    # WHEN / THEN
    with pytest.raises(ValueError):
        _ = SyncSolrClient(base_url=input_url, request_timeout_sec=input_timeout)
