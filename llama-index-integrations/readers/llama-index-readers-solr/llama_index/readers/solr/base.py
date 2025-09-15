"""
Solr reader over REST api.
"""

from typing import Any, Optional

import pysolr

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class SolrReader(BasePydanticReader):
    """
    Read documents from a Solr index.

    These documents can then be used in a downstream Llama Index data structure.
    """

    endpoint: str = Field(description="Full endpoint, including collection info.")
    _client: Any = PrivateAttr()

    def __init__(
        self,
        endpoint: str,
    ):
        """Initialize with parameters."""
        super().__init__(endpoint=endpoint)
        self._client = pysolr.Solr(endpoint)

    def load_data(
        self,
        query: dict[str, Any],
        field: str,
        id_field: str = "id",
        metadata_fields: Optional[list[str]] = None,
        embedding: Optional[str] = None,
    ) -> list[Document]:
        r"""
        Read data from the Solr index. At least one field argument must be specified.

        Args:
            query (dict): The Solr query parameters.
                - "q" is required.
                - "rows" should be specified or will default to 10 by Solr.
                - If "fl" is provided, it is respected exactly as given.
                  If "fl" is NOT provided, a default `fl` is constructed from
                  {id_field, field, embedding?, metadata_fields?}.
            field (str): Field in Solr to retrieve as document text.
            id_field (str): Field in Solr to retrieve as the document identifier. Defaults to "id".
            metadata_fields (list[str], optional): Fields to include as metadata. Defaults to None.
            embedding (str, optional): Field to use for embeddings. Defaults to None.

        Raises:
            ValueError: If the HTTP call to Solr fails.

        Returns:
            list[Document]: A list of retrieved documents where field is populated.

        """
        if "q" not in query:
            raise ValueError("Query parameters must include a 'q' field for the query.")

        fl_default = {}
        if "fl" not in query:
            fields = [id_field, field]
            if embedding:
                fields.append(embedding)
            if metadata_fields:
                fields.extend(metadata_fields)
            fl_default = {"fl": ",".join(fields)}

        try:
            query_params = {
                **query,
                **fl_default,
            }
            results = self._client.search(**query_params)
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Failed to query Solr endpoint: {e!s}") from e

        documents: list[Document] = []
        for doc in results.docs:
            if field not in doc:
                continue

            doc_kwargs: dict[str, Any] = {
                "id_": str(doc[id_field]),
                "text": doc[field],
                **({"embedding": doc.get(embedding)} if embedding else {}),
                "metadata": {
                    metadata_field: doc[metadata_field]
                    for metadata_field in (metadata_fields or [])
                    if metadata_field in doc
                },
            }
            documents.append(Document(**doc_kwargs))
        return documents
