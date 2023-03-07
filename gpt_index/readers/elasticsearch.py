"""Elasticsearch (or Opensearch) reader over REST api.

This only uses the basic search api, so it will work with Elasticsearch and Opensearch.

"""


from typing import List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class ElasticsearchReader(BaseReader):
    """
    Read documents from an Elasticsearch/Opensearch index.

    These documents can then be used in a downstream Llama Index data structure.

    Args:
        endpoint (str): URL (http/https) of cluster
        index (str): Name of the index (required)
        httpx_client_args (dict): Optional additional args to pass to the `httpx.Client`
    """

    def __init__(
        self, endpoint: str, index: str, httpx_client_args: Optional[dict] = None
    ):
        """Initialize with parameters."""
        import_err_msg = """
            `httpx` package not found. Install via `pip install httpx`
        """
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)
        self._client = httpx.Client(base_url=endpoint, **(httpx_client_args or {}))
        self._index = index
        self._endpoint = endpoint

    def load_data(
        self,
        field: str,
        query: Optional[dict] = None,
        embedding_field: Optional[str] = None,
    ) -> List[Document]:
        """Read data from the Elasticsearch index.

        Args:
            field (str): Field in the document to retrieve text from
            query (Optional[dict]): Elasticsearch JSON query DSL object.
                For example:
                {"query": {"match": {"message": {"query": "this is a test"}}}}
            embedding_field (Optional[str]): If there are embeddings stored in
                this index, this field can be used
                to set the embedding field on the returned Document list.
        Returns:
            List[Document]: A list of documents.

        """
        res = self._client.post(f"{self._index}/_search", json=query).json()
        documents = []
        for hit in res["hits"]["hits"]:
            value = hit["_source"][field]
            embedding = hit["_source"].get(embedding_field or "", None)
            documents.append(
                Document(text=value, extra_info=hit["_source"], embedding=embedding)
            )
        return documents
