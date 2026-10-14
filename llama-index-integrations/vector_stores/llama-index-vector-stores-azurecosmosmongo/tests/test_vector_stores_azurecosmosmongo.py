from unittest.mock import MagicMock

import pytest
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.azurecosmosmongo import AzureCosmosDBMongoDBVectorSearch
from llama_index.vector_stores.azuredocumentdb import AzureDocumentDBVectorSearch


def test_class():
    names_of_base_classes = [
        b.__name__ for b in AzureCosmosDBMongoDBVectorSearch.__mro__
    ]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes
    assert issubclass(AzureCosmosDBMongoDBVectorSearch, AzureDocumentDBVectorSearch)


def test_deprecation_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        AzureCosmosDBMongoDBVectorSearch,
        "_create_vector_search_index",
        lambda self: None,
    )

    with pytest.warns(DeprecationWarning, match="AzureDocumentDBVectorSearch"):
        AzureCosmosDBMongoDBVectorSearch(mongodb_client=MagicMock())
