from typing import Any, Dict, List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class MetalReader(BaseReader):
    """Metal reader.

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
            import metal_sdk  # noqa: F401
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
        id_to_text_map: Dict[str, str],
        query_str: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        separate_documents: bool = True,
        **query_kwargs: Any
    ) -> List[Document]:
        """Load data from Metal.

        Args:
            id_to_text_map (Dict[str, str]): A map from ID's to text.
            query_str (str): Query string for text search.
            top_k (int): Number of results to return.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search.
            separate_documents (Optional[bool]): Whether to return separate
                documents per retrieved entry. Defaults to True.
            **query_kwargs: Keyword arguments to pass to the search.

        Returns:
            List[Document]: A list of documents.
        """

        payload = {
            "text": query_str,
            "filters": filters,
        }
        response = self.metal_client.search(payload, limit=top_k, **query_kwargs)

        documents = []
        for item in response["results"]:
            if item["id"] not in id_to_text_map:
                raise ValueError("ID not found in id_to_text_map.")
            text = id_to_text_map[item["id"]]
            documents.append(Document(text=text))

        if not separate_documents:
            text_list = [doc.get_text() for doc in documents]
            text = "\n\n".join(text_list)
            documents = [Document(text=text)]

        return documents
