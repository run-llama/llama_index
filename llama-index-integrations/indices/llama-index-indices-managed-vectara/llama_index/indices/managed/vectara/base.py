"""
Managed index.

A managed Index - where the index is accessible via some API that
interfaces a managed service.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from hashlib import blake2b
from typing import Any, List, Optional, Sequence, Type

import requests
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.data_structs.data_structs import IndexDict, IndexStructType
from llama_index.core.indices.managed.base import BaseManagedIndex, IndexType
from llama_index.core.llms.utils import LLMType, resolve_llm
from llama_index.core.schema import (
    BaseNode,
    Document,
    Node,
    MediaResource,
    TransformComponent,
)
from llama_index.core.settings import Settings
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer


_logger = logging.getLogger(__name__)


class VectaraIndexStruct(IndexDict):
    """Vectara Index Struct."""

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get index struct type."""
        return IndexStructType.VECTARA


class VectaraIndex(BaseManagedIndex):
    """
    Vectara Index.

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
        vectara_corpus_key: Optional[str] = None,
        vectara_api_key: Optional[str] = None,
        use_core_api: bool = False,
        parallelize_ingest: bool = False,
        x_source_str: str = "llama_index",
        **kwargs: Any,
    ) -> None:
        """Initialize the Vectara API."""
        self.parallelize_ingest = parallelize_ingest
        index_struct = VectaraIndexStruct(
            index_id=str(vectara_corpus_key),
            summary="Vectara Index",
        )

        super().__init__(
            show_progress=show_progress,
            index_struct=index_struct,
            **kwargs,
        )

        self._vectara_corpus_key = vectara_corpus_key or str(
            os.environ.get("VECTARA_CORPUS_KEY")
        )

        self._vectara_api_key = vectara_api_key or os.environ.get("VECTARA_API_KEY")
        if self._vectara_corpus_key is None or self._vectara_api_key is None:
            _logger.warning(
                "Can't find Vectara credentials or corpus_key in environment."
            )
            raise ValueError("Missing Vectara credentials")
        else:
            _logger.debug(f"Using corpus key {self._vectara_corpus_key}")

        # identifies usage source for internal measurement
        self._x_source_str = x_source_str

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
            self._build_index_from_nodes(nodes, self.use_core_api)

    def _build_index_from_nodes(
        self, nodes: Sequence[BaseNode], use_core_api: bool = False
    ) -> IndexDict:
        docs = [
            Document(
                text_resource=MediaResource(
                    text=node.get_content()
                ),  # may need to add get_content().
                metadata=node.metadata,  # type: ignore
                id_=node.id_,  # type: ignore
            )
            for node in nodes
        ]
        self.add_documents(docs, use_core_api)
        return self.index_struct

    def _get_corpus_key(self, corpus_key: str) -> str:
        """
        Get the corpus key to use for the index.
        If corpus_key is provided, check if it is one of the valid corpus keys.
        If not, use the first corpus key in the list.
        """
        if corpus_key is not None:
            if corpus_key in self._vectara_corpus_key.split(","):
                return corpus_key
        return self._vectara_corpus_key.split(",")[0]

    def _get_post_headers(self) -> dict:
        """Returns headers that should be attached to each post request."""
        return {
            "x-api-key": self._vectara_api_key,
            "Content-Type": "application/json",
            "X-Source": self._x_source_str,
        }

    def _delete_doc(self, doc_id: str, corpus_key: Optional[str] = None) -> bool:
        """
        Delete a document from the Vectara corpus.

        Args:
            doc_id (str): ID of the document to delete.
            corpus_key (str): corpus key to delete the document from.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        valid_corpus_key = self._get_corpus_key(corpus_key)
        body = {}
        response = self._session.delete(
            f"https://api.vectara.io/v2/corpora/{valid_corpus_key}/documents/{doc_id}",
            data=json.dumps(body),
            verify=True,
            headers=self._get_post_headers(),
            timeout=self.vectara_api_timeout,
        )

        if response.status_code != 204:
            _logger.error(
                f"Delete request failed for doc_id = {doc_id} with status code "
                f"{response.status_code}, text {response.json()['messages'][0]}"
            )
            return False
        return True

    # THE WAY THAT DOCUMENTS ARE INDEXED NOW IS VERY DIFFERENT THAN BEFORE, SO WE WILL NEED TO RESTRUCTURE HOW THIS IS DONE (SEE API PLAYGROUND)
    # DIFFERENCE IN TWO IMPLEMENTATION STYLES IS SPECIFIED BY PARAMETER `use_core_api` (check where it is used in process of creating documents).
    def _index_doc(self, doc: dict, corpus_key) -> str:
        response = self._session.post(
            headers=self._get_post_headers(),
            url=f"https://api.vectara.io/v2/corpora/{corpus_key}/documents",
            data=json.dumps(doc),
            timeout=self.vectara_api_timeout,
            verify=True,
        )

        status_code = response.status_code
        if status_code == 201:
            return "E_SUCCEEDED"

        result = response.json()
        return result["messages"][0]

    def _insert(
        self,
        nodes: Sequence[BaseNode],
        corpus_key: Optional[str] = None,
        use_core_api: bool = False,
        **insert_kwargs: Any,
    ) -> None:
        """Insert a set of documents (each a node)."""

        def gen_hash(s: str) -> str:
            hash_object = blake2b(digest_size=32)
            hash_object.update(s.encode("utf-8"))
            return hash_object.hexdigest()

        docs = []
        for node in nodes:
            metadata = node.metadata.copy()
            metadata["framework"] = "llama_index"  # NOT SURE WHAT THIS IS FOR
            section_key = "document_parts" if use_core_api else "sections"
            text = node.get_content()

            # NEED TO FIGURE OUT A WAY TO LET PEOPLE CHOOSE THEIR OWN DOC ID
            # IF THEY DON'T SPECIFY ONE, THEN A RANDOM ONE IS CREATED AUTOMATICALLY.
            # THIS IS PROBLEMATIC BECAUSE THEN THE SAME DOCUMENT CAN BE INGESTED TWICE.
            doc_id = gen_hash(text) if len(node.node_id) == 36 else node.node_id
            doc = {
                "id": doc_id,
                "type": "core" if use_core_api else "structured",
                "metadata": node.metadata,
                section_key: [{"text": text}],
            }
            docs.append(doc)

        valid_corpus_key = self._get_corpus_key(corpus_key)
        if self.parallelize_ingest:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._index_doc, doc, valid_corpus_key)
                    for doc in docs
                ]
                for future in futures:
                    ecode = future.result()
                    if ecode != "E_SUCCEEDED":
                        _logger.error(
                            f"Error indexing document in Vectara with error code {ecode}"
                        )
            self.doc_ids.extend([doc["id"] for doc in docs])
        else:
            for doc in docs:
                ecode = self._index_doc(doc, valid_corpus_key)
                if ecode != "E_SUCCEEDED":
                    _logger.error(
                        f"Error indexing document in Vectara with error code {ecode}"
                    )
                self.doc_ids.append(doc["id"])

    def add_documents(
        self,
        docs: Sequence[Document],
        corpus_key: Optional[str],
        use_core_api: bool = False,
        allow_update: bool = True,
    ) -> None:
        nodes = [
            Node(
                text_resource=MediaResource(text=doc.text),
                metadata=doc.metadata,
                id_=doc.doc_id,
            )
            for doc in docs
        ]
        self._insert(nodes, corpus_key, use_core_api)

    def insert_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
        chunking_strategy: Optional[dict] = None,
        table_extraction_config: Optional[dict] = None,
        filename: Optional[str] = None,
        corpus_key: Optional[str] = None,
        **insert_kwargs: Any,
    ) -> Optional[str]:
        """
        Vectara provides a way to add files (binary or text) directly via our API
        where pre-processing and chunking occurs internally in an optimal way
        This method provides a way to use that API in Llama_index.

        # ruff: noqa: E501
        Full API Docs: https://docs.vectara.com/docs/rest-api/upload-file

        Args:
            file_path: local file path
                Files could be text, HTML, PDF, markdown, doc/docx, ppt/pptx, etc.
                see API docs for full list
            metadata: Optional dict of metadata associated with the file
            chunking_strategy: Optional dict specifying max number of characters per chunk
            table_extraction_config: Optional dict specifying whether or not to extract tables from document
            filename: Optional string specifying the filename


        Returns:
            List of ids associated with each of the files indexed
        """
        if not os.path.exists(file_path):
            _logger.error(f"File {file_path} does not exist")
            return None

        if filename is None:
            filename = file_path

        files = {"file": (filename, open(file_path, "rb"))}

        if metadata:
            metadata["framework"] = "llama_index"
            files["metadata"] = (None, json.dumps(metadata), "application/json")

        if chunking_strategy:
            files["chunking_strategy"] = (
                None,
                json.dumps(chunking_strategy),
                "application/json",
            )

        if table_extraction_config:
            files["table_extraction_config"] = (
                None,
                json.dumps(table_extraction_config),
                "application/json",
            )

        headers = self._get_post_headers()
        headers.pop("Content-Type")
        valid_corpus_key = self._get_corpus_key(corpus_key)
        response = self._session.post(
            f"https://api.vectara.io/v2/corpora/{valid_corpus_key}/upload_file",
            files=files,
            verify=True,
            headers=headers,
            timeout=self.vectara_api_timeout,
        )

        res = response.json()
        if response.status_code == 201:
            doc_id = res["id"]
            self.doc_ids.append(doc_id)
            return doc_id
        elif response.status_code == 400:
            _logger.info(f"File upload failed with error message {res['field_errors']}")
            return None
        else:
            _logger.info(f"File upload failed with error message {res['messages'][0]}")
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
        from llama_index.indices.managed.vectara.retriever import (
            VectaraRetriever,
        )

        return VectaraRetriever(self, **kwargs)

    def as_chat_engine(self, **kwargs: Any) -> BaseChatEngine:
        kwargs["summary_enabled"] = True
        retriever = self.as_retriever(**kwargs)
        kwargs.pop("summary_enabled")
        from llama_index.indices.managed.vectara.query import (
            VectaraChatEngine,
        )

        return VectaraChatEngine.from_args(retriever, **kwargs)  # type: ignore

    def as_query_engine(
        self, llm: Optional[LLMType] = None, **kwargs: Any
    ) -> BaseQueryEngine:
        if kwargs.get("summary_enabled", True):
            from llama_index.indices.managed.vectara.query import (
                VectaraQueryEngine,
            )

            kwargs["summary_enabled"] = True
            retriever = self.as_retriever(**kwargs)
            return VectaraQueryEngine.from_args(retriever=retriever, **kwargs)  # type: ignore
        else:
            from llama_index.core.query_engine.retriever_query_engine import (
                RetrieverQueryEngine,
            )

            llm = (
                resolve_llm(llm, callback_manager=self._callback_manager)
                or Settings.llm
            )

            retriever = self.as_retriever(**kwargs)
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT,
                llm=llm,
            )
            return RetrieverQueryEngine.from_args(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                **kwargs,
            )

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        show_progress: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        **kwargs: Any,
    ) -> IndexType:
        """Build a Vectara index from a sequence of documents."""
        nodes = [
            Node(
                text_resource=MediaResource(text=document.text),
                metadata=document.metadata,
                id_=document.doc_id,
            )
            for document in documents
        ]

        return cls(
            nodes=nodes,
            show_progress=show_progress,
            **kwargs,
        )
