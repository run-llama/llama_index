"""Vectara Vector store index.

An index that that is built on top of an existing vector store.

"""

import logging
import os
from typing import Any, List, Optional, Sequence, Dict
import requests
import json

from llama_index.schema import TextNode
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.schema import Document

logger = logging.getLogger(__name__)


class VectaraVectorStore(VectorStore):
    """Vectara Vector Store.

    This is vector store that uses Vectara as the backend.

    During query time, the index uses Vectara to query for the top
    k most similar nodes.

    Args:
        vectara_customer_id: Vectara's customer ID
        vectara_corpus_id: the ID of the corpus in Vectara
        vectara_api_key: Vectara's API key
    """

    stores_text: bool = True

    def __init__(
        self,
        vectara_customer_id: Optional[str] = None,
        vectara_corpus_id: Optional[str] = None,
        vectara_api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with Vectara API."""
        self._vectara_customer_id = vectara_customer_id or os.environ.get(
            "VECTARA_CUSTOMER_ID"
        )
        self._vectara_corpus_id = vectara_corpus_id or os.environ.get(
            "VECTARA_CORPUS_ID"
        )
        self._vectara_api_key = vectara_api_key or os.environ.get("VECTARA_API_KEY")
        if (
            self._vectara_customer_id is None
            or self._vectara_corpus_id is None
            or self._vectara_api_key is None
        ):
            logger.warning(
                "Can't find Vectara credentials, customer_id or corpus_id in "
                "environment."
            )
        else:
            logger.debug(f"Using corpus id {self._vectara_corpus_id}")

        self._session = requests.Session()  # to reuse connections
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        self._session.mount("https://", adapter)
        self.vectara_api_timeout = 60

    def _get_post_headers(self) -> dict:
        """Returns headers that should be attached to each post request."""
        return {
            "x-api-key": self._vectara_api_key,
            "customer-id": self._vectara_customer_id,
            "Content-Type": "application/json",
        }

    def _delete_doc(self, doc_id: str) -> bool:
        """
        Delete a document from the Vectara corpus.

        Args:
            url (str): URL of the page to delete.
            doc_id (str): ID of the document to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        body = {
            "customer_id": self._vectara_customer_id,
            "corpus_id": self._vectara_corpus_id,
            "document_id": doc_id,
        }
        response = self._session.post(
            "https://api.vectara.io/v1/delete-doc",
            data=json.dumps(body),
            verify=True,
            headers=self._get_post_headers(),
            timeout=self.vectara_api_timeout,
        )

        if response.status_code != 200:
            logger.error(
                f"Delete request failed for doc_id = {doc_id} with status code "
                f"{response.status_code}, reason {response.reason}, text "
                f"{response.text}"
            )
            return False
        return True

    def _index_doc(self, doc: dict) -> str:
        request: dict[str, Any] = {}
        request["customer_id"] = self._vectara_customer_id
        request["corpus_id"] = self._vectara_corpus_id
        request["document"] = doc

        response = self._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/index",
            data=json.dumps(request),
            timeout=self.vectara_api_timeout,
            verify=True,
        )

        status_code = response.status_code

        result = response.json()

        status_str = result["status"]["code"] if "status" in result else None
        if status_code == 409 or status_str and (status_str == "ALREADY_EXISTS"):
            return "E_ALREADY_EXISTS"
        elif status_str and (status_str == "FORBIDDEN"):
            return "E_NO_PERMISSIONS"
        else:
            return "E_SUCCEEDED"

    def add_documents(
        self,
        docs: List[Document],
    ) -> List[str]:
        """Add texts to Vectara Index
        Note: in the case of Vectara, embedding is ignored since Vectara generates its own embedding.

        Args
            texts: List[str]: list of text strings to add to Vectara
        """

        ids = []
        for doc in docs:
            doc_id = doc.id_
            doc_metadata = doc.metadata.copy()
            doc_metadata["source"] = "llamaindex"
            doc = {
                "document_id": doc_id,
                "metadataJson": json.dumps(doc_metadata),
                "section": [{"text": doc.text}],
            }
            success_str = self._index_doc(doc)
            if success_str == "E_ALREADY_EXISTS":
                print(
                    "Document already exists in Vectara corpus - deleting and reindexing"
                )
                self._delete_doc(doc_id)
                self._index_doc(doc)
            elif success_str == "E_NO_PERMISSIONS":
                print(
                    """No permissions to add document to Vectara. 
                    Check your corpus ID, customer ID and API key"""
                )
            ids.append(doc_id)

        return ids

    def delete(self, doc_id: str) -> None:
        """
        Delete document with doc_id.

        Args:
            doc_id (str): The doc_id of the document to delete.
        """
        # delete by filtering on the doc_id metadata
        self._delete_doc(doc_id)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query: VectorStoreQuery which includes the query_str, similarity_top_k, alpha (for hybrid search) and filters
        """

        filters = (
            " and ".join([f"doc.{k} = {v}" for k, v in query.filters.items()])
            if query.filters
            else ""
        )
        data = json.dumps(
            {
                "query": [
                    {
                        "query": query.query_str,
                        "start": 0,
                        "num_results": query.similarity_top_k,
                        "context_config": {
                            "sentences_before": 2,
                            "sentences_after": 2,
                        },
                        "corpus_key": [
                            {
                                "customer_id": self._vectara_customer_id,
                                "corpus_id": self._vectara_corpus_id,
                                "metadataFilter": filters,
                                "lexical_interpolation_config": {
                                    "lambda": query.alpha if query.alpha else 0.025
                                },
                            }
                        ],
                    }
                ]
            }
        )

        response = self._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/query",
            data=data,
            timeout=self.vectara_api_timeout,
        )

        if response.status_code != 200:
            logger.error(
                "Query failed %s",
                f"(code {response.status_code}, reason {response.reason}, details "
                f"{response.text})",
            )
            return []

        result = response.json()
        responses = result["responseSet"][0]["response"]
        docs = result["responseSet"][0]["document"]
        vectara_default_metadata = ["lang", "len", "offset"]

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        for x in responses:
            md = [m for m in x["metadata"] if m["name"] not in vectara_default_metadata]
            doc_inx = x["documentIndex"]
            doc_id = docs[doc_inx]["id"]
            node = TextNode(text=x["text"], id_=doc_id, metadata=md)
            top_k_nodes.append(node)
            top_k_ids.append(doc_id)
            top_k_scores.append(x["score"])

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
