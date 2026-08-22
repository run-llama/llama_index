import logging
import asyncio
from typing import Any, List, Optional, cast

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)

class FeedoVectorStore(BasePydanticVectorStore):
    """
    Feedo Vector Store.
    
    Provides decentralized, encrypted at rest vector storage via the Feedo Protocol.
    
    Args:
        usage_key (str): The Feedo usage key for authentication.
        did (str): The decentralized identity (DID) for the agent.
        namespace (Optional[str]): Tenant isolation namespace (e.g., room_id or user_id).
    """

    stores_text: bool = True
    flat_metadata: bool = False

    _client: Any = PrivateAttr()
    _namespace: str = PrivateAttr()

    def __init__(
        self,
        usage_key: str,
        did: str,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        try:
            from feedo.router import NodeRouter
            from feedo.modules.search import SearchModule
        except ImportError:
            raise ImportError(
                "Could not import feedo python package. "
                "Please install it with `pip install feedo-sdk`."
            )
        
        router = NodeRouter()
        self._client = SearchModule(router=router, usage_key=usage_key, did=did)
        self._namespace = namespace or ""

    @property
    def client(self) -> Any:
        return self._client

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to index."""
        # Using a new event loop or the current one to run async methods
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        ids = []
        for node in nodes:
            metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )
            hash_id = node.id_
            text = node.get_content(metadata_mode=MetadataMode.NONE) or ""
            
            loop.run_until_complete(
                self._client.index_private_document(
                    hash_id=hash_id,
                    plaintext=text,
                    metadata=metadata,
                    namespace=self._namespace
                )
            )
            ids.append(hash_id)
            
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using ref_doc_id.
        Currently, Feedo supports deleting by namespace. 
        """
        raise NotImplementedError("Delete by ref_doc_id not supported by Feedo natively yet.")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        query_str = query.query_str
        if not query_str:
            raise ValueError("query_str is required for Feedo query.")

        limit = query.similarity_top_k
        
        response = loop.run_until_complete(
            self._client.search(
                query=query_str,
                limit=limit,
                namespace=self._namespace
            )
        )
        
        documents = response.get("documents", []) or response.get("data", []) or response.get("results", [])
        
        nodes = []
        similarities = []
        ids = []

        for doc in documents:
            text = doc.get("text", "") or doc.get("content", "")
            metadata = doc.get("metadata", {})
            node_id = doc.get("hash_id") or metadata.get("id") or metadata.get("doc_id")
            
            try:
                node = metadata_dict_to_node(metadata, text=text)
            except Exception:
                node = TextNode(text=text, metadata=metadata, id_=node_id or "unknown")
            
            nodes.append(node)
            similarities.append(float(doc.get("score", 0.0)))
            ids.append(node_id or "unknown")

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
