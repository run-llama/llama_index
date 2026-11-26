import warnings
from typing import Any, Dict, Optional

from llama_index.vector_stores.azuredocumentdb import AzureDocumentDBVectorSearch


class AzureCosmosDBMongoDBVectorSearch(AzureDocumentDBVectorSearch):
    """Deprecated compatibility wrapper for AzureDocumentDBVectorSearch."""

    def __init__(
        self,
        mongodb_client: Optional[Any] = None,
        db_name: str = "default_db",
        collection_name: str = "default_collection",
        index_name: str = "default_vector_search_index",
        id_key: str = "id",
        embedding_key: str = "content_vector",
        text_key: str = "text",
        metadata_key: str = "metadata",
        cosmos_search_kwargs: Optional[Dict] = None,
        insert_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            "AzureCosmosDBMongoDBVectorSearch is deprecated; use "
            "AzureDocumentDBVectorSearch from llama_index.vector_stores.azuredocumentdb instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            mongodb_client=mongodb_client,
            db_name=db_name,
            collection_name=collection_name,
            index_name=index_name,
            id_key=id_key,
            embedding_key=embedding_key,
            text_key=text_key,
            metadata_key=metadata_key,
            cosmos_search_kwargs=cosmos_search_kwargs,
            insert_kwargs=insert_kwargs,
            **kwargs,
        )
