from unittest.mock import MagicMock

import pytest
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.azuredocumentdb import AzureDocumentDBVectorSearch


def test_class():
    names_of_base_classes = [b.__name__ for b in AzureDocumentDBVectorSearch.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_legacy_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    mongo_client = MagicMock()
    monkeypatch.delenv("AZURE_DOCUMENTDB_URI", raising=False)
    monkeypatch.setenv("AZURE_COSMOSDB_MONGODB_URI", "mongodb://legacy.example")
    monkeypatch.setattr(
        "llama_index.vector_stores.azuredocumentdb.base.pymongo.MongoClient",
        MagicMock(return_value=mongo_client),
    )
    monkeypatch.setattr(
        AzureDocumentDBVectorSearch,
        "_create_vector_search_index",
        lambda self: None,
    )

    with pytest.warns(DeprecationWarning, match="AZURE_COSMOSDB_MONGODB_URI"):
        AzureDocumentDBVectorSearch()
