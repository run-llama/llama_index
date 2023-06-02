"""ChatGPT Plugin vector store."""

import os
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm.auto import tqdm

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)


def convert_docs_to_json(embedding_results: List[NodeWithEmbedding]) -> List[Dict]:
    """Convert docs to JSON."""
    docs = []
    for embedding_result in embedding_results:
        # TODO: add information for other fields as well
        # fields taken from
        # https://rb.gy/nmac9u
        doc_dict = {
            "id": embedding_result.id,
            "text": embedding_result.node.get_text(),
            # NOTE: this is the doc_id to reference document
            "source_id": embedding_result.ref_doc_id,
            # "url": "...",
            # "created_at": ...,
            # "author": "..."",
        }
        extra_info = embedding_result.node.extra_info
        if extra_info is not None:
            if "source" in extra_info:
                doc_dict["source"] = extra_info["source"]
            if "source_id" in extra_info:
                doc_dict["source_id"] = extra_info["source_id"]
            if "url" in extra_info:
                doc_dict["url"] = extra_info["url"]
            if "created_at" in extra_info:
                doc_dict["created_at"] = extra_info["created_at"]
            if "author" in extra_info:
                doc_dict["author"] = extra_info["author"]

        docs.append(doc_dict)
    return docs


class ChatGPTRetrievalPluginClient(VectorStore):
    """ChatGPT Retrieval Plugin Client.

    In this client, we make use of the endpoints defined by ChatGPT.

    Args:
        endpoint_url (str): URL of the ChatGPT Retrieval Plugin.
        bearer_token (Optional[str]): Bearer token for the ChatGPT Retrieval Plugin.
        retries (Optional[Retry]): Retry object for the ChatGPT Retrieval Plugin.
        batch_size (int): Batch size for the ChatGPT Retrieval Plugin.
    """

    stores_text: bool = True
    is_embedding_query: bool = False

    def __init__(
        self,
        endpoint_url: str,
        bearer_token: Optional[str] = None,
        retries: Optional[Retry] = None,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._endpoint_url = endpoint_url
        self._bearer_token = bearer_token or os.getenv("BEARER_TOKEN")
        self._retries = retries
        self._batch_size = batch_size

        self._s = requests.Session()
        self._s.mount("http://", HTTPAdapter(max_retries=self._retries))

    @property
    def client(self) -> None:
        """Get client."""
        return None

    def add(
        self,
        embedding_results: List[NodeWithEmbedding],
    ) -> List[str]:
        """Add embedding_results to index."""
        headers = {"Authorization": f"Bearer {self._bearer_token}"}

        docs_to_upload = convert_docs_to_json(embedding_results)
        for i in tqdm(range(0, len(docs_to_upload), self._batch_size)):
            i_end = min(i + self._batch_size, len(docs_to_upload))
            self._s.post(
                f"{self._endpoint_url}/upsert",
                headers=headers,
                json={"documents": docs_to_upload[i:i_end]},
            )

        return [result.id for result in embedding_results]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        headers = {"Authorization": f"Bearer {self._bearer_token}"}
        self._s.post(
            f"{self._endpoint_url}/delete",
            headers=headers,
            json={"ids": [ref_doc_id]},
        )

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for ChatGPT Plugin yet.")

        if query.query_str is None:
            raise ValueError("query_str must be provided")
        headers = {"Authorization": f"Bearer {self._bearer_token}"}
        # TODO: add metadata filter
        queries = [{"query": query.query_str, "top_k": query.similarity_top_k}]
        res = requests.post(
            f"{self._endpoint_url}/query", headers=headers, json={"queries": queries}
        )

        nodes = []
        similarities = []
        ids = []
        for query_result in res.json()["results"]:
            for result in query_result["results"]:
                result_id = result["id"]
                result_txt = result["text"]
                result_score = result["score"]
                result_ref_doc_id = result["source_id"]
                node = Node(
                    doc_id=result_id,
                    text=result_txt,
                    relationships={DocumentRelationship.SOURCE: result_ref_doc_id},
                )
                nodes.append(node)
                similarities.append(result_score)
                ids.append(result_id)

            # NOTE: there should only be one query
            break

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
