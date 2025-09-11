import types
import pytest
import pysolr

from llama_index.core.readers.base import BaseReader
from llama_index.readers.solr import SolrReader


@pytest.fixture(scope="module")
def dummy_endpoint() -> str:
    return "http://localhost:8983/solr/collection1"


@pytest.fixture
def dummy_solr(mocker) -> pysolr.Solr:
    ctor = mocker.patch("llama_index.readers.solr.base.pysolr.Solr", autospec=True)
    return ctor.return_value


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in SolrReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_initialization(dummy_solr, dummy_endpoint) -> None:
    reader = SolrReader(endpoint=dummy_endpoint)
    assert reader._client is dummy_solr


def test_load_data_happy_path_with_metadata_and_embedding(
    dummy_solr, dummy_endpoint
) -> None:
    dummy_solr.search.return_value = types.SimpleNamespace(
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
        query={"q": "*:*", "rows": 10, "fl": "ignored"},
        field="content_t",
        metadata_fields=["title_t"],
        embedding="vec",
    )

    assert len(docs) == 1
    doc = docs[0]
    assert doc.id_ == "1"
    assert doc.get_content() == "hello world"
    assert doc.embedding == [0.1, 0.2]
    assert doc.metadata == {"title_t": "Title"}
    dummy_solr.search.assert_called_once()
    assert dummy_solr.search.call_args.kwargs["fl"] == "id,content_t,vec,title_t"


def test_load_data_skips_docs_without_required_field(
    dummy_solr, dummy_endpoint
) -> None:
    dummy_solr.search.return_value = types.SimpleNamespace(
        docs=[
            {"id": "1", "title_t": "has title only"},  # missing content_t
            {"id": "2", "content_t": "kept"},
        ]
    )

    reader = SolrReader(endpoint=dummy_endpoint)
    docs = reader.load_data(query={"q": "*:*"}, field="content_t")
    assert [d.id_ for d in docs] == ["2"]
    assert docs[0].get_content() == "kept"


def test_load_data_raises_when_q_missing(dummy_solr, dummy_endpoint) -> None:
    reader = SolrReader(endpoint=dummy_endpoint)
    with pytest.raises(ValueError):
        _ = reader.load_data(query={}, field="content_t")
