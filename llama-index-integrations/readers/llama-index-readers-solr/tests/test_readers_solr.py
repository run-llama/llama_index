import types
import pytest
import pysolr

from llama_index.core.readers.base import BaseReader
from llama_index.readers.solr import SolrReader


@pytest.fixture(scope="module")
def dummy_endpoint() -> str:
    return "http://localhost:8983/solr/collection1"


@pytest.fixture
def mock_solr(mocker) -> pysolr.Solr:
    ctor = mocker.patch("llama_index.readers.solr.base.pysolr.Solr", autospec=True)
    return ctor.return_value


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in SolrReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_initialization(mock_solr, dummy_endpoint) -> None:
    reader = SolrReader(endpoint=dummy_endpoint)
    assert reader._client is mock_solr


def test_load_data_builds_default_fl_and_returns_docs(
    mock_solr, dummy_endpoint
) -> None:
    mock_solr.search.return_value = types.SimpleNamespace(
        docs=[
            {
                "id": "1",
                "content_t": "hello world",
                "title_t": "Title",
                "vec": [0.1, 0.2],
            }
        ]
    )

    reader = SolrReader(endpoint=dummy_endpoint)
    docs = reader.load_data(
        query={"q": "*:*", "rows": 10, "fl": "respected"},
        field="content_t",
        metadata_fields=["title_t"],
        embedding="vec",
    )

    mock_solr.search.assert_called_once()
    assert mock_solr.search.call_args.kwargs["fl"] == "respected"

    assert len(docs) == 1
    doc = docs[0]
    assert doc.id_ == "1"
    assert doc.get_content() == "hello world"
    assert doc.embedding == [0.1, 0.2]
    assert doc.metadata == {"title_t": "Title"}


def test_load_data_constructs_fl_when_missing_and_skips_bad_docs(
    mock_solr, dummy_endpoint
) -> None:
    mock_solr.search.return_value = types.SimpleNamespace(
        docs=[
            {
                "id": "1",
                "title_t": "has title only",
            },  # missing content_t, expected to be skipped
            {"id": "2", "content_t": "kept"},
        ]
    )

    reader = SolrReader(endpoint=dummy_endpoint)
    docs = reader.load_data(query={"q": "*:*"}, field="content_t")

    called = mock_solr.search.call_args.kwargs
    assert called["fl"] == "id,content_t"

    assert [d.id_ for d in docs] == ["2"]
    assert docs[0].get_content() == "kept"

    # Defaults
    assert docs[0].embedding is None
    assert docs[0].metadata == {}


def test_load_data_custom_id_field_and_numeric_coercion(
    mock_solr, dummy_endpoint
) -> None:
    mock_solr.search.return_value = types.SimpleNamespace(
        docs=[
            {
                "my_id": 1234567890123,  # long-ish numeric id
                "body_s": "num id keeps working",
                "x": "meta",
            }
        ]
    )

    reader = SolrReader(endpoint=dummy_endpoint)
    docs = reader.load_data(
        query={"q": "*:*"},
        field="body_s",
        id_field="my_id",
        metadata_fields=["x"],
    )

    called = mock_solr.search.call_args.kwargs
    assert called["fl"] == "my_id,body_s,x"  # custom my_id field

    assert len(docs) == 1
    d = docs[0]
    assert d.id_ == "1234567890123"  # coerced to str
    assert d.get_content() == "num id keeps working"
    assert d.metadata == {"x": "meta"}
    assert d.embedding is None


def test_load_data_raises_when_q_missing(mock_solr, dummy_endpoint) -> None:
    reader = SolrReader(endpoint=dummy_endpoint)
    with pytest.raises(ValueError):
        _ = reader.load_data(query={}, field="content_t")
