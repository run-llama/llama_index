"""Pathway Retriever."""

import json
from typing import List, Optional

import requests

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.schema import (
    NodeWithScore,
    QueryBundle,
    TextNode,
)


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
        responses = response.json()
        return sorted(responses, key=lambda x: x["dist"])

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


class PathwayRetriever(BaseRetriever):
    """
    Pathway retriever.

    Pathway is an open data processing framework.
    It allows you to easily develop data transformation pipelines
    that work with live data sources and changing data.

    This is the client that implements Retriever API for PathwayVectorServer.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Initializing the Pathway retriever client."""
        self.client = _VectorStoreClient(host, port, url)
        self.similarity_top_k = similarity_top_k
        super().__init__(callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        rets = self.client(query=query_bundle.query_str, k=self.similarity_top_k)
        items = [
            NodeWithScore(
                node=TextNode(text=ret["text"], extra_info=ret["metadata"]),
                # Transform cosine distance into a similairty score
                # (higher is more similar)
                score=1 - ret["dist"],
            )
            for ret in rets
        ]
        return sorted(items, key=lambda x: x.score or 0.0, reverse=True)
