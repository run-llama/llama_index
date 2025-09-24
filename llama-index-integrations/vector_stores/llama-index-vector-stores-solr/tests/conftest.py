from collections.abc import Iterator
from datetime import date, datetime, timezone

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc

from typing import Any, Optional, Union
from uuid import uuid4
from zoneinfo import ZoneInfo

import aiosolr
import numpy as np
import pysolr
import pytest
import pytest_docker
import requests

from llama_index.vector_stores.solr.client import (
    SolrResponseHeader,
    SolrSelectResponse,
    SolrUpdateResponse,
)
from llama_index.vector_stores.solr.client.responses import SolrSelectResponseBody


@pytest.fixture
def mock_base_solr_url() -> str:
    return "https://some-cloud.com/api/solr/some-collection"


@pytest.fixture
def mock_solr_raw_input_documents() -> list[dict[str, Any]]:
    return [
        {
            "id": "node1",
            "text_txt_en": "some content 1",
            "boolean_b": True,
            "bytes_s": b"\xe4\xb8\xad\xe6\x96\x87",
            "date_dt": date(2025, 8, 1),
            "datetime_dt": datetime(2025, 9, 1, 1, 2, 3, tzinfo=UTC),
            "float_f": 0.1,
            "int_i": 1,
            "ndarray_fs": np.array([0.1] * 64),
            "np_float_f": np.float64(0.1),
            "np_int_i": np.int_(1),
            "string_list_ss": ["tag1", "tag2"],
        },
        {
            "id": "node2",
            "text_txt_en": "some content 2",
            "boolean_b": False,
            "bytes_s": b"\xe0\xae\xa4\xe0\xae\xae\xe0\xae\xbf\xe0\xae\xb4\xe0\xaf\x8d",
            "date_dt": date(2025, 8, 2),
            "datetime_dt": datetime(
                2025, 9, 2, 2, 3, 4, tzinfo=ZoneInfo("America/New_York")
            ),
            "float_f": 0.2,
            "int_i": 2,
            "ndarray_fs": np.array([0.2] * 64),
            "np_float_f": np.float64(0.2),
            "np_int_i": np.int_(2),
            "string_list_ss": ["tag3"],
        },
        {
            "id": "node3",
            "text_txt_en": "some content 3",
            "boolean_b": True,
            "bytes_s": b"\xd5\xb0\xd5\xa1\xd5\xb5\xd5\xa5\xd6\x80\xd5\xa5\xd5\xb6",
            "date_dt": date(2025, 8, 3),
            "datetime_dt": datetime(2025, 9, 3, 2, 3, 4, tzinfo=None),
            "float_f": 0.3,
            "int_i": 3,
            "ndarray_fs": np.array([0.3] * 64),
            "np_float_f": np.float64(0.3),
            "np_int_i": np.int_(3),
            "string_list_ss": [],
        },
    ]


@pytest.fixture
def mock_solr_updated_input_documents() -> list[dict[str, Any]]:
    return [
        {
            "id": "node1",
            "text_txt_en": "some content 1",
            "boolean_b": True,
            "bytes_s": "中文",
            "date_dt": "2025-08-01T00:00:00Z",
            "datetime_dt": "2025-09-01T01:02:03Z",
            "float_f": 0.1,
            "int_i": 1,
            "ndarray_fs": [0.1] * 64,
            "np_float_f": 0.1,
            "np_int_i": 1,
            "string_list_ss": ["tag1", "tag2"],
        },
        {
            "id": "node2",
            "text_txt_en": "some content 2",
            "boolean_b": False,
            "bytes_s": "தமிழ்",
            "date_dt": "2025-08-02T00:00:00Z",
            "datetime_dt": "2025-09-02T06:03:04Z",
            "float_f": 0.2,
            "int_i": 2,
            "ndarray_fs": [0.2] * 64,
            "np_float_f": 0.2,
            "np_int_i": 2,
            "string_list_ss": ["tag3"],
        },
        {
            "id": "node3",
            "text_txt_en": "some content 3",
            "boolean_b": True,
            "bytes_s": "հայերեն",
            "date_dt": "2025-08-03T00:00:00Z",
            "datetime_dt": "2025-09-03T02:03:04Z",
            "float_f": 0.3,
            "int_i": 3,
            "ndarray_fs": [0.3] * 64,
            "np_float_f": 0.3,
            "np_int_i": 3,
            "string_list_ss": [],
        },
    ]


@pytest.fixture
def mock_solr_expected_retrieved_documents(
    mock_solr_updated_input_documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out_docs: list[dict[str, Any]] = []
    for doc in mock_solr_updated_input_documents:
        # at retrieval time, Solr won't return null, the dictionary will just be missing
        # the value altogether
        out_docs.append(
            {
                key: value
                for key, value in doc.items()
                if value is not None and value != []
            }
        )
    return out_docs


@pytest.fixture
def mock_solr_update_response() -> SolrUpdateResponse:
    return SolrUpdateResponse(
        response_header=SolrResponseHeader(status=200, QTime=None),
    )


@pytest.fixture
def mock_pysolr_update_response(
    mock_solr_update_response: SolrUpdateResponse,
) -> str:
    # pysolr returns the raw JSON from Solr
    return mock_solr_update_response.model_dump_json(exclude_none=True, by_alias=True)


@pytest.fixture
def mock_solr_delete_response_xml() -> str:
    return '<?xml version="1.0" encoding="UTF-8"?>\n<response>\n\n<lst name="responseHeader">\n  <int name="rf">1</int>\n  <int name="status">0</int>\n  <int name="QTime">25</int>\n</lst>\n</response>\n'


@pytest.fixture
def mock_solr_delete_response() -> SolrUpdateResponse:
    return SolrUpdateResponse(
        response_header=SolrResponseHeader(rf=1, status=0, QTime=25), debug=None
    )


@pytest.fixture
def mock_aiosolr_update_response(
    mock_solr_update_response: SolrUpdateResponse,
) -> aiosolr.Response:
    # aiosolr wraps responses in a custom object
    return aiosolr.Response(
        data=mock_solr_update_response.model_dump(by_alias=True),
        status=200,
    )


@pytest.fixture
def mock_solr_select_response(
    mock_solr_updated_input_documents: list[dict[str, Any]],
) -> SolrSelectResponse:
    return SolrSelectResponse(
        response=SolrSelectResponseBody(
            docs=mock_solr_updated_input_documents,
            num_found=len(mock_solr_updated_input_documents),
            num_found_exact=True,
            start=200,
        ),
        response_header=SolrResponseHeader(status=200),
    )


@pytest.fixture
def mock_pysolr_search_results(
    mock_solr_select_response: SolrSelectResponse,
) -> pysolr.Results:
    # pysolr wraps results in a custom object
    return pysolr.Results(mock_solr_select_response.model_dump(by_alias=True))


@pytest.fixture
def mock_aiosolr_search_results(
    mock_solr_select_response: SolrSelectResponse,
) -> aiosolr.Response:
    # aiosolr wraps results in a custom object
    return aiosolr.Response(
        data=mock_solr_select_response.model_dump(by_alias=True),
        status=200,
    )


@pytest.fixture(scope="session")
def docker_compose_project_name() -> str:
    # Pin the project name to avoid creating multiple stacks
    # see docs: https://github.com/avast/pytest-docker
    return "llama-index-vector-stores-solr"


@pytest.fixture(scope="session")
def docker_setup():
    # Stop the stack before starting a new one
    # see docs: https://github.com/avast/pytest-docker
    return ["down -v", "up --build -d"]


def is_responsive(url: str) -> bool:
    try:
        # Test basic connectivity first
        response = requests.get(url)
        if response.status_code != 200:
            return False

        # Test Solr admin API to ensure it's fully ready
        admin_response = requests.get(f"{url}/solr/admin/cores?action=STATUS")
        return admin_response.status_code == 200
    except (
        ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.RequestException,
    ):
        return False


@pytest.fixture(scope="session")
def docker_solr_url(
    docker_ip: Union[str, Any], docker_services: pytest_docker.Services
) -> str:
    """Ensure the ``solr-local`` container is up and responsive."""
    port = docker_services.port_for("solr-local", 8983)
    url = f"http://{docker_ip}:{port}"
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive(url)
    )
    return url


def list_collections(base_solr_url: str) -> dict[str, dict[str, Any]]:
    """List all collections in a Solr instance."""
    # ref: https://solr.apache.org/guide/solr/latest/deployment-guide/collection-management.html#list
    resp = requests.get(f"{base_solr_url}/solr/admin/cores?action=STATUS")
    assert resp.status_code == 200, (
        f"Failed to list collections, code={resp.status_code}: {resp.text}"
    )
    collections: dict[str, dict[str, Any]] = resp.json()["status"]
    return collections


def create_collection(
    base_solr_url: str, collection_name: str, config_set: str = "_default"
) -> None:
    """Create a collection in a Solr instance."""
    # ref: https://solr.apache.org/guide/solr/latest/deployment-guide/collection-management.html#create
    resp = requests.get(
        f"{base_solr_url}/solr/admin/collections?action=CREATE"
        f"&name={collection_name}"
        f"&collection.configName={config_set}"
        f"&numShards=1"
    )
    assert resp.status_code == 200, (
        f"Failed to create collection='{collection_name}', "
        f"code={resp.status_code}: {resp.text}"
    )


def delete_collection(base_solr_url: str, collection_name: str) -> None:
    """Delete a collection in a Solr instance."""
    # ref: https://solr.apache.org/guide/solr/latest/deployment-guide/collection-management.html#delete
    resp = requests.get(
        f"{base_solr_url}/solr/admin/collections?action=DELETE&name={collection_name}"
    )
    assert resp.status_code == 200, (
        f"Could not delete collection='{collection_name}', "
        f"code={resp.status_code}: {resp.text}"
    )
    collections = list_collections(base_solr_url)
    assert collection_name not in collections


def get_collection_url(
    base_solr_url: str,
    collection_name: Optional[str] = None,
    config_set: str = "_default",
) -> tuple[str, str]:
    """Get a collection URL for the given name, creating it if needed."""
    collection_name = collection_name or uuid4().hex
    all_collections = list_collections(base_solr_url)
    if collection_name not in all_collections:
        create_collection(base_solr_url, collection_name, config_set)
    collection_url = f"{base_solr_url}/solr/{collection_name}"
    return collection_name, collection_url


@pytest.fixture()
def function_unique_solr_collection_url(docker_solr_url: str) -> Iterator[str]:
    """
    Build a unique Solr collection that lives for one test.

    This fixture returns a local Solr URL that can be used by various clients
    """
    # build a unique collection
    collection_name, collection_url = get_collection_url(docker_solr_url)

    yield collection_url

    # remove the collection
    delete_collection(docker_solr_url, collection_name)


@pytest.fixture()
def function_unique_solr_with_knn_collection_url(
    function_unique_solr_collection_url: str,
) -> str:
    """
    Add KNN schema to an existing Solr collection for vector search testing.
    """
    base_url, collection_name = function_unique_solr_collection_url.rsplit(
        "/", maxsplit=1
    )
    schema_url = f"{base_url}/{collection_name}/schema"

    # Check if vector field already exists
    field_check = requests.get(f"{schema_url}/fields/vector_field")

    if field_check.status_code != 200:
        # Field doesn't exist, add the schema
        schema_update = {
            "add-field-type": {
                "name": "knn_vector_64",
                "class": "solr.DenseVectorField",
                "vectorDimension": 64,
                "similarityFunction": "cosine",
                "knnAlgorithm": "hnsw",
            },
            "add-field": {
                "name": "vector_field",
                "type": "knn_vector_64",
                "indexed": True,
                "stored": True,
            },
        }

        resp = requests.post(
            schema_url, json=schema_update, headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 200, (
            f"Failed to add KNN schema, code={resp.status_code}: {resp.text}"
        )

    return function_unique_solr_collection_url

    # The collection cleanup is handled by the parent fixture


def compare_documents(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    sort_key: str = "id",
    required_keys: Optional[list[str]] = None,
) -> None:
    required_keys = required_keys or list(expected[0].keys())
    expected_sorted = sorted(expected, key=lambda d: d[sort_key])
    actual_sorted = [
        {k: v for k, v in doc.items() if k in required_keys}
        for doc in sorted(actual, key=lambda d: d[sort_key])
    ]
    assert actual_sorted == expected_sorted


# shared parameters for integration tests
params_search_queries = pytest.mark.parametrize(
    ("input_query", "expected_doc_indexes", "requires_score"),
    [
        ({"q": "*:*"}, [0, 1, 2], False),
        ({"q": 'text_txt_en:"some content 2"'}, [1], False),
        ({"q": 'text_txt_en:"some content 2"', "fl": "*,score"}, [1], True),
    ],
    ids=[
        "Select all documents (no score)",
        "Search on text field (no score)",
        "Search on text field (with score)",
    ],
)

params_delete_by_id = pytest.mark.parametrize(
    ("ids_to_delete", "expected_remaining_ids"),
    [
        (["node1"], ["node2", "node3"]),
        (["node1", "node3"], ["node2"]),
        (["node1", "node2", "node3"], []),
    ],
    ids=[
        "delete=1, remain=2",
        "delete=2, remain=1",
        "delete=all, remain=0",
    ],
)
