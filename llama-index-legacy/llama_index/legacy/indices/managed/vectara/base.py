"""Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.

"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from hashlib import blake2b
from typing import Any, Dict, List, Optional, Sequence, Type

import requests

from llama_index.legacy.core.base_query_engine import BaseQueryEngine
from llama_index.legacy.core.base_retriever import BaseRetriever
from llama_index.legacy.data_structs.data_structs import IndexDict, IndexStructType
from llama_index.legacy.indices.managed.base import BaseManagedIndex, IndexType
from llama_index.legacy.schema import BaseNode, Document, MetadataMode, TextNode
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.storage.storage_context import StorageContext

_logger = logging.getLogger(__name__)


class VectaraIndexStruct(IndexDict):
    """Vectara Index Struct."""

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get index struct type."""
        return IndexStructType.VECTARA


class VectaraIndex(BaseManagedIndex):
    """Vectara Index.

    The Vectara index implements a managed index that uses Vectara as the backend.
    Vectara performs a lot of the functions in traditional indexes in the backend:
    - breaks down a document into chunks (nodes)
    - Creates the embedding for each chunk (node)
    - Performs the search for the top k most similar nodes to a query
    - Optionally can perform summarization of the top k nodes

    Args:
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.

    """

    def __init__(
        self,
        show_progress: bool = False,
        nodes: Optional[Sequence[BaseNode]] = None,
        vectara_customer_id: Optional[str] = None,
        vectara_corpus_id: Optional[str] = None,
        vectara_api_key: Optional[str] = None,
        use_core_api: bool = False,
        parallelize_ingest: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Vectara API."""
        self.parallelize_ingest = parallelize_ingest
        index_struct = VectaraIndexStruct(
            index_id=str(vectara_corpus_id),
            summary="Vectara Index",
        )

        super().__init__(
            show_progress=show_progress,
            index_struct=index_struct,
            service_context=ServiceContext.from_defaults(
                llm=None, llm_predictor=None, embed_model=None
            ),
            **kwargs,
        )
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
            _logger.warning(
                "Can't find Vectara credentials, customer_id or corpus_id in "
                "environment."
            )
            raise ValueError("Missing Vectara credentials")
        else:
            _logger.debug(f"Using corpus id {self._vectara_corpus_id}")

        # setup requests session with max 3 retries and 90s timeout
        # for calling Vectara API
        self._session = requests.Session()  # to reuse connections
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        self._session.mount("https://", adapter)
        self.vectara_api_timeout = 90
        self.use_core_api = use_core_api
        self.doc_ids: List[str] = []

        # if nodes is specified, consider each node as a single document
        # and use _build_index_from_nodes() to add them to the index
        if nodes is not None:
            self._build_index_from_nodes(nodes, use_core_api)

    def _build_index_from_nodes(
        self, nodes: Sequence[BaseNode], use_core_api: bool = False
    ) -> IndexDict:
        docs = [
            Document(
                text=node.get_content(metadata_mode=MetadataMode.NONE),
                metadata=node.metadata,  # type: ignore
                id_=node.id_,  # type: ignore
            )
            for node in nodes
        ]
        self.add_documents(docs, use_core_api)
        return self.index_struct

    def _get_post_headers(self) -> dict:
        """Returns headers that should be attached to each post request."""
        return {
            "x-api-key": self._vectara_api_key,
            "customer-id": self._vectara_customer_id,
            "Content-Type": "application/json",
            "X-Source": "llama_index",
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
            "customerId": self._vectara_customer_id,
            "corpusId": self._vectara_corpus_id,
            "documentId": doc_id,
        }
        response = self._session.post(
            "https://api.vectara.io/v1/delete-doc",
            data=json.dumps(body),
            verify=True,
            headers=self._get_post_headers(),
            timeout=self.vectara_api_timeout,
        )

        if response.status_code != 200:
            _logger.error(
                f"Delete request failed for doc_id = {doc_id} with status code "
                f"{response.status_code}, reason {response.reason}, text "
                f"{response.text}"
            )
            return False
        return True

    def _index_doc(self, doc: dict) -> str:
        request: Dict[str, Any] = {}
        request["customerId"] = self._vectara_customer_id
        request["corpusId"] = self._vectara_corpus_id
        request["document"] = doc

        if "parts" in doc:
            api_url = "https://api.vectara.io/v1/core/index"
        else:
            api_url = "https://api.vectara.io/v1/index"

        response = self._session.post(
            headers=self._get_post_headers(),
            url=api_url,
            data=json.dumps(request),
            timeout=self.vectara_api_timeout,
            verify=True,
        )

        status_code = response.status_code

        result = response.json()

        status_str = result["status"]["code"] if "status" in result else None
        if status_code == 409 or status_str and (status_str == "ALREADY_EXISTS"):
            return "E_ALREADY_EXISTS"
        elif status_code == 200 or status_str and (status_str == "INVALID_ARGUMENT"):
            return "E_INVALID_ARGUMENT"
        elif status_str and (status_str == "FORBIDDEN"):
            return "E_NO_PERMISSIONS"
        else:
            return "E_SUCCEEDED"

    def _insert(
        self,
        nodes: Sequence[BaseNode],
        use_core_api: bool = False,
        **insert_kwargs: Any,
    ) -> None:
        """Insert a set of documents (each a node)."""

        def gen_hash(s: str) -> str:
            hash_object = blake2b()
            hash_object.update(s.encode("utf-8"))
            return hash_object.hexdigest()

        docs = []
        for node in nodes:
            metadata = node.metadata.copy()
            metadata["framework"] = "llama_index"
            section_key = "parts" if use_core_api else "section"
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            doc_id = gen_hash(text)
            doc = {
                "documentId": doc_id,
                "metadataJson": json.dumps(node.metadata),
                section_key: [{"text": text}],
            }
            docs.append(doc)

        if self.parallelize_ingest:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._index_doc, doc) for doc in docs]
                for future in futures:
                    ecode = future.result()
                    if ecode != "E_SUCCEEDED":
                        _logger.error(
                            f"Error indexing document in Vectara with error code {ecode}"
                        )
        else:
            for doc in docs:
                ecode = self._index_doc(doc)
                if ecode != "E_SUCCEEDED":
                    _logger.error(
                        f"Error indexing document in Vectara with error code {ecode}"
                    )
                self.doc_ids.append(doc_id)

    def add_documents(
        self,
        docs: Sequence[Document],
        use_core_api: bool = False,
        allow_update: bool = True,
    ) -> None:
        nodes = [
            TextNode(text=doc.get_content(), metadata=doc.metadata) for doc in docs  # type: ignore
        ]
        self._insert(nodes, use_core_api)

    def insert_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
        **insert_kwargs: Any,
    ) -> Optional[str]:
        """Vectara provides a way to add files (binary or text) directly via our API
        where pre-processing and chunking occurs internally in an optimal way
        This method provides a way to use that API in Llama_index.

        # ruff: noqa: E501
        Full API Docs: https://docs.vectara.com/docs/api-reference/indexing-apis/
        file-upload/file-upload-filetypes

        Args:
            file_path: local file path
                Files could be text, HTML, PDF, markdown, doc/docx, ppt/pptx, etc.
                see API docs for full list
            metadata: Optional list of metadata associated with the file

        Returns:
            List of ids associated with each of the files indexed
        """
        if not os.path.exists(file_path):
            _logger.error(f"File {file_path} does not exist")
            return None

        metadata = metadata or {}
        metadata["framework"] = "llama_index"
        files: dict = {
            "file": (file_path, open(file_path, "rb")),
            "doc_metadata": json.dumps(metadata),
        }
        headers = self._get_post_headers()
        headers.pop("Content-Type")
        response = self._session.post(
            f"https://api.vectara.io/upload?c={self._vectara_customer_id}&o={self._vectara_corpus_id}&d=True",
            files=files,
            verify=True,
            headers=headers,
            timeout=self.vectara_api_timeout,
        )

        if response.status_code == 409:
            doc_id = response.json()["document"]["documentId"]
            _logger.info(
                f"File {file_path} already exists on Vectara "
                f"(doc_id={doc_id}), skipping"
            )
            return None
        elif response.status_code == 200:
            return response.json()["document"]["documentId"]
        else:
            _logger.info(f"Error indexing file {file_path}: {response.json()}")
            return None

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        raise NotImplementedError(
            "Vectara does not support deleting a reference document"
        )

    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        raise NotImplementedError(
            "Vectara does not support updating a reference document"
        )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a Retriever for this managed index."""
        from llama_index.legacy.indices.managed.vectara.retriever import (
            VectaraRetriever,
        )

        return VectaraRetriever(self, **kwargs)

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        if kwargs.get("summary_enabled", True):
            from llama_index.legacy.indices.managed.vectara.query import (
                VectaraQueryEngine,
            )

            kwargs["summary_enabled"] = True
            retriever = self.as_retriever(**kwargs)
            return VectaraQueryEngine.from_args(retriever, **kwargs)  # type: ignore
        else:
            from llama_index.legacy.query_engine.retriever_query_engine import (
                RetrieverQueryEngine,
            )

            kwargs["retriever"] = self.as_retriever(**kwargs)
            return RetrieverQueryEngine.from_args(**kwargs)

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None,
        service_context: Optional[ServiceContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> IndexType:
        """Build a Vectara index from a sequence of documents."""
        nodes = [
            TextNode(text=document.get_content(), metadata=document.metadata)  # type: ignore
            for document in documents
        ]
        return cls(
            nodes=nodes,
            show_progress=show_progress,
            **kwargs,
        )
