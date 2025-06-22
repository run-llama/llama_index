from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class MetalReader(BaseReader):
    """
    Metal reader.

    Args:
        api_key (str): Metal API key.
        client_id (str): Metal client ID.
        index_id (str): Metal index ID.

    """

    def __init__(self, api_key: str, client_id: str, index_id: str):
        import_err_msg = (
            "`metal_sdk` package not found, please run `pip install metal_sdk`"
        )
        try:
            import metal_sdk  # noqa
        except ImportError:
            raise ImportError(import_err_msg)
        from metal_sdk.metal import Metal

        """Initialize with parameters."""
        self._api_key = api_key
        self._client_id = client_id
        self._index_id = index_id
        self.metal_client = Metal(api_key, client_id, index_id)

    def load_data(
        self,
        limit: int,
        query_embedding: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        separate_documents: bool = True,
        **query_kwargs: Any,
    ) -> List[Document]:
        """
        Load data from Metal.

        Args:
            query_embedding (Optional[List[float]]): Query embedding for search.
            limit (int): Number of results to return.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search.
            separate_documents (Optional[bool]): Whether to return separate
                documents per retrieved entry. Defaults to True.
            **query_kwargs: Keyword arguments to pass to the search.

        Returns:
            List[Document]: A list of documents.

        """
        payload = {
            "embedding": query_embedding,
            "filters": filters,
        }
        response = self.metal_client.search(payload, limit=limit, **query_kwargs)

        documents = []
        for item in response["data"]:
            text = item["text"] or (item["metadata"] and item["metadata"]["text"])
            documents.append(Document(text=text))

        if not separate_documents:
            text_list = [doc.get_content() for doc in documents]
            text = "\n\n".join(text_list)
            documents = [Document(text=text)]

        return documents
