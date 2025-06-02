"""Pathway reader."""

import json
from typing import List, Optional

import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


# Copied from https://github.com/pathwaycom/pathway/blob/main/python/pathway/xpacks/llm/vector_store.py
# to remove dependency on Pathway library when only the client is used.
class _VectorStoreClient:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
    ):
        """
        A client you can use to query :py:class:`VectorStoreServer`.

        Please provide either the `url`, or `host` and `port`.

        Args:
            - host: host on which `:py:class:`VectorStoreServer` listens
            - port: port on which `:py:class:`VectorStoreServer` listens
            - url: url at which `:py:class:`VectorStoreServer` listens

        """
        err = "Either (`host` and `port`) or `url` must be provided, but not both."
        if url is not None:
            if host or port:
                raise ValueError(err)
            self.url = url
        else:
            if host is None:
                raise ValueError(err)
            port = port or 80
            self.url = f"http://{host}:{port}"

    def query(
        self, query: str, k: int = 3, metadata_filter: Optional[str] = None
    ) -> List[dict]:
        """
        Perform a query to the vector store and fetch results.

        Args:
            - query:
            - k: number of documents to be returned
            - metadata_filter: optional string representing the metadata filtering query
                in the JMESPath format. The search will happen only for documents
                satisfying this filtering.

        """
        data = {"query": query, "k": k}
        if metadata_filter is not None:
            data["metadata_filter"] = metadata_filter
        url = self.url + "/v1/retrieve"
        response = requests.post(
            url,
            data=json.dumps(data),
            headers={"Content-Type": "application/json"},
            timeout=3,
        )
        return response.json()

    # Make an alias
    __call__ = query

    def get_vectorstore_statistics(self) -> dict:
        """Fetch basic statistics about the vector store."""
        url = self.url + "/v1/statistics"
        response = requests.post(
            url,
            json={},
            headers={"Content-Type": "application/json"},
        )
        return response.json()

    def get_input_files(
        self,
        metadata_filter: Optional[str] = None,
        filepath_globpattern: Optional[str] = None,
    ) -> list:
        """
        Fetch information on documents in the vector store.

        Args:
            metadata_filter: optional string representing the metadata filtering query
                in the JMESPath format. The search will happen only for documents
                satisfying this filtering.
            filepath_globpattern: optional glob pattern specifying which documents
                will be searched for this query.

        """
        url = self.url + "/v1/inputs"
        response = requests.post(
            url,
            json={
                "metadata_filter": metadata_filter,
                "filepath_globpattern": filepath_globpattern,
            },
            headers={"Content-Type": "application/json"},
        )
        return response.json()


class PathwayReader(BaseReader):
    """
    Pathway reader.

    Retrieve documents from Pathway data indexing pipeline.

    Args:
        host (str): The URI where Pathway is currently hosted.
        port (str | int): The port number on which Pathway is listening.

    See Also:
        llamaindex.retriever.pathway.PathwayRetriever and,
        llamaindex.retriever.pathway.PathwayVectorServer

    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
    ):
        """Initializing the Pathway reader client."""
        self.client = _VectorStoreClient(host, port, url)

    def load_data(
        self,
        query_text: str,
        k: Optional[int] = 4,
        metadata_filter: Optional[str] = None,
    ) -> List[Document]:
        """
        Load data from Pathway.

        Args:
            query_text (str): The text to get the closest neighbors of.
            k (int): Number of results to return.
            metadata_filter (str): Filter to be applied.

        Returns:
            List[Document]: A list of documents.

        """
        results = self.client(query_text, k, metadata_filter)
        documents = []
        for return_elem in results:
            document = Document(
                text=return_elem["text"],
                extra_info=return_elem["metadata"],
            )

            documents.append(document)

        return documents
