"""
Elasticsearch (or Opensearch) reader over REST api.

This only uses the basic search api, so it will work with Elasticsearch and Opensearch.

"""

from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class ElasticsearchReader(BasePydanticReader):
    """
    Read documents from an Elasticsearch/Opensearch index.

    These documents can then be used in a downstream Llama Index data structure.

    Args:
        endpoint (str): URL (http/https) of cluster
        index (str): Name of the index (required)
        httpx_client_args (dict): Optional additional args to pass to the `httpx.Client`

    """

    is_remote: bool = True
    endpoint: str
    index: str
    httpx_client_args: Optional[dict] = None

    _client: Any = PrivateAttr()

    def __init__(
        self, endpoint: str, index: str, httpx_client_args: Optional[dict] = None
    ):
        """Initialize with parameters."""
        super().__init__(
            endpoint=endpoint, index=index, httpx_client_args=httpx_client_args
        )
        import_err_msg = """
            `httpx` package not found. Install via `pip install httpx`
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(import_err_msg)
        self._client = httpx.Client(base_url=endpoint, **(httpx_client_args or {}))

    @classmethod
    def class_name(cls) -> str:
        return "ElasticsearchReader"

    def load_data(
        self,
        field: str,
        query: Optional[dict] = None,
        embedding_field: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Read data from the Elasticsearch index.

        Args:
            field (str): Field in the document to retrieve text from
            query (Optional[dict]): Elasticsearch JSON query DSL object.
                For example:
                {"query": {"match": {"message": {"query": "this is a test"}}}}
            embedding_field (Optional[str]): If there are embeddings stored in
                this index, this field can be used
                to set the embedding field on the returned Document list.
            metadata_fields (Optional[List[str]]): Fields used as metadata. Default
                is all fields in the document except those specified by the
                field and embedding_field parameters.

        Returns:
            List[Document]: A list of documents.

        """
        res = self._client.post(f"{self.index}/_search", json=query).json()
        documents = []
        for hit in res["hits"]["hits"]:
            doc_id = hit["_id"]
            value = hit["_source"][field]
            embedding = hit["_source"].get(embedding_field or "", None)
            if metadata_fields:
                metadata = {
                    k: v for k, v in hit["_source"].items() if k in metadata_fields
                }
            else:
                hit["_source"].pop(field)
                hit["_source"].pop(embedding_field or "", None)
                metadata = hit["_source"]
            documents.append(
                Document(id_=doc_id, text=value, metadata=metadata, embedding=embedding)
            )
        return documents
