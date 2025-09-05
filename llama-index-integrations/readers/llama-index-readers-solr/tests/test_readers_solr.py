from llama_index.core.readers.base import BaseReader
from llama_index.readers.solr import SolrReader


def test_class():
    names_of_base_classes = [b.__name__ for b in SolrReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_initialization() -> None:
    """Initialization."""
    reader = SolrReader(endpoint="http://localhost:8983/solr/collection1")
    assert reader._client is not None
