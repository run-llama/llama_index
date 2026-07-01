"""
Dakera vector store index.

Dakera is a self-hosted, decay-weighted vector memory server that handles
embedding generation internally. This integration wraps Dakera's REST API
so it can be used as a drop-in LlamaIndex VectorStore.

Quick-start:
    docker run -p 3300:3300 -e DAKERA_API_KEY=demo ghcr.io/dakera-ai/dakera:latest

Usage:
    from llama_index.vector_stores.dakera import DakeraVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext

    store = DakeraVectorStore(
        base_url="http://localhost:3300",
        agent_id="my-agent",
        api_key="demo",            # optional — omit if auth is disabled
    )
    storage_context = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import httpx
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
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

_DEFAULT_BASE_URL = "http://localhost:3300"
_DEFAULT_TIMEOUT = 30.0


class DakeraVectorStore(BasePydanticVectorStore):
    """
    Dakera Vector Store.

    Dakera is a decay-weighted, self-hosted vector memory server. Embedding
    is performed server-side, so no local embedding model is required.

    The store maps each LlamaIndex node to a Dakera memory record:
    - ``node.text`` → ``content`` (the raw text sent for embedding)
    - ``node.node_id`` → stored in ``tags`` as ``node_id=<id>`` for later retrieval
    - Node metadata (incl. the full ``_node_content`` JSON blob written by
      :func:`node_to_metadata_dict`) → stored in the Dakera ``tags`` list

    Because Dakera's ``/v1/memory/search`` endpoint returns plain text (not
    embeddings), ``is_embedding_query`` is set to ``False``. LlamaIndex will
    pass ``query_str`` rather than requiring a pre-computed ``query_embedding``.

    Args:
        base_url: Base URL of the Dakera server (default: ``http://localhost:3300``).
        agent_id: Agent namespace used to isolate memories.
        session_id: Optional session scope for memory storage.
        api_key: Optional bearer token for authenticated Dakera deployments.
        top_k: Default number of results to return per search (default: 10).
        importance: Optional default importance score (0.0–1.0) written at
            store time. Overridden by ``node.metadata.get("importance")``.
        timeout: HTTP request timeout in seconds (default: 30).

    Examples:
        `pip install llama-index-vector-stores-dakera`

        ```python
        from llama_index.vector_stores.dakera import DakeraVectorStore

        store = DakeraVectorStore(
            base_url="http://localhost:3300",
            agent_id="my-agent",
            api_key="demo",
        )
        ```

    """

    stores_text: bool = True
    is_embedding_query: bool = False  # Dakera embeds server-side; query by text

    # Public Pydantic fields (serialisable)
    base_url: str
    agent_id: str
    session_id: Optional[str]
    top_k: int
    importance: Optional[float]
    timeout: float

    # Private — not serialised
    _client: httpx.Client = PrivateAttr()
    _async_client: httpx.AsyncClient = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "DakeraVectorStore"

    @property
    def client(self) -> httpx.Client:
        """Return the underlying synchronous HTTP client."""
        return self._client

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        agent_id: str = "default",
        session_id: Optional[str] = None,
        api_key: Optional[str] = None,
        top_k: int = 10,
        importance: Optional[float] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(
            base_url=base_url.rstrip("/"),
            agent_id=agent_id,
            session_id=session_id,
            top_k=top_k,
            importance=importance,
            timeout=timeout,
        )
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )
        self._async_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_store_payload(self, node: BaseNode) -> Dict[str, Any]:
        """Build the ``/v1/memory/store`` request body for a single node."""
        metadata = node_to_metadata_dict(node)
        # Serialise entire metadata blob to a JSON string stored as one tag
        # so it can be recovered on retrieval via metadata_dict_to_node.
        meta_tag = f"_llama_meta={json.dumps(metadata, ensure_ascii=False)}"

        tags: List[str] = [f"node_id={node.node_id}", meta_tag]

        # Propagate user-supplied tags from node metadata
        user_tags = node.metadata.get("tags", [])
        if isinstance(user_tags, list):
            tags.extend(str(t) for t in user_tags)

        payload: Dict[str, Any] = {
            "content": node.get_content(metadata_mode="none") or node.node_id,
            "agent_id": self.agent_id,
            "tags": tags,
        }
        if self.session_id:
            payload["session_id"] = self.session_id

        # Importance: prefer per-node override, fall back to store-level default
        node_importance = node.metadata.get("importance", self.importance)
        if node_importance is not None:
            payload["importance"] = float(node_importance)

        return payload

    @staticmethod
    def _memory_to_node(memory: Dict[str, Any]) -> BaseNode:
        """Reconstruct a LlamaIndex :class:`BaseNode` from a Dakera memory record."""
        tags: List[str] = memory.get("tags") or []

        # Find the serialised LlamaIndex metadata blob
        meta_json: Optional[str] = None
        for tag in tags:
            if tag.startswith("_llama_meta="):
                meta_json = tag[len("_llama_meta="):]
                break

        if meta_json:
            try:
                meta_dict = json.loads(meta_json)
                return metadata_dict_to_node(meta_dict)
            except Exception:
                logger.warning(
                    "Failed to deserialise _llama_meta tag for memory %s; "
                    "falling back to TextNode.",
                    memory.get("id"),
                )

        # Fallback: plain TextNode from content
        return TextNode(
            text=memory.get("content", ""),
            id_=memory.get("id", ""),
        )

    # ------------------------------------------------------------------
    # VectorStore interface — synchronous
    # ------------------------------------------------------------------

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Store nodes in Dakera.

        Each node is stored as a separate memory record. Dakera handles
        embedding server-side; no ``node.embedding`` is required.

        Args:
            nodes: Sequence of :class:`BaseNode` objects to store.
            **add_kwargs: Unused; present for interface compatibility.

        Returns:
            List of Dakera memory IDs (one per node).

        """
        ids: List[str] = []
        for node in nodes:
            payload = self._build_store_payload(node)
            try:
                resp = self._client.post("/v1/memory/store", json=payload)
                resp.raise_for_status()
                memory_id: str = resp.json()["memory"]["id"]
                ids.append(memory_id)
            except Exception as exc:
                logger.error(
                    "Dakera store failed for node %s: %s", node.node_id, exc
                )
                raise
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete all memories whose ``node_id`` tag matches *ref_doc_id*.

        Dakera's ``/v1/memory/forget`` accepts a list of explicit memory IDs.
        Because nodes are stored with a ``node_id=<id>`` tag we first query
        Dakera for that tag to locate the memory ID(s), then delete them.

        Args:
            ref_doc_id: The node / doc ID to remove.
            **delete_kwargs: Accepts ``memory_ids`` (list[str]) to bypass the
                lookup and delete by Dakera memory ID directly.

        """
        memory_ids: Optional[List[str]] = delete_kwargs.get("memory_ids")

        if memory_ids is None:
            # Search to find the memory IDs associated with this node
            try:
                resp = self._client.post(
                    "/v1/memory/search",
                    json={
                        "agent_id": self.agent_id,
                        "query": f"node_id={ref_doc_id}",
                        "top_k": 100,
                    },
                )
                resp.raise_for_status()
                hits = resp.json().get("memories", [])
                memory_ids = [
                    h["memory"]["id"]
                    for h in hits
                    if any(
                        t == f"node_id={ref_doc_id}"
                        for t in (h["memory"].get("tags") or [])
                    )
                ]
            except Exception as exc:
                logger.error(
                    "Dakera search-before-delete failed for %s: %s", ref_doc_id, exc
                )
                raise

        if not memory_ids:
            logger.debug("No Dakera memories found for ref_doc_id=%s", ref_doc_id)
            return

        try:
            resp = self._client.post(
                "/v1/memory/forget",
                json={"agent_id": self.agent_id, "memory_ids": memory_ids},
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.error(
                "Dakera forget failed for %s: %s", memory_ids, exc
            )
            raise

    def query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Semantic search over Dakera memories.

        Because ``is_embedding_query = False``, LlamaIndex populates
        ``query.query_str`` with the raw question text. Dakera embeds it
        server-side and returns scored memories.

        Args:
            query: :class:`VectorStoreQuery` — ``query_str`` is used; any
                pre-computed ``query_embedding`` is silently ignored because
                Dakera's API does not accept raw embedding vectors.
            **kwargs: Unused; present for interface compatibility.

        Returns:
            :class:`VectorStoreQueryResult` with nodes, similarity scores, and IDs.

        """
        if not query.query_str:
            raise ValueError(
                "DakeraVectorStore requires query.query_str (text). "
                "Pre-computed embeddings are not supported — Dakera embeds server-side."
            )

        top_k = query.similarity_top_k or self.top_k
        payload: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "query": query.query_str,
            "top_k": top_k,
        }
        if self.session_id:
            payload["session_id"] = self.session_id

        try:
            resp = self._client.post("/v1/memory/search", json=payload)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("Dakera search failed: %s", exc)
            raise

        hits = resp.json().get("memories", [])
        nodes: List[BaseNode] = []
        scores: List[float] = []
        ids: List[str] = []

        for hit in hits:
            memory = hit["memory"]
            score: float = float(hit.get("score", 0.0))
            node = self._memory_to_node(memory)
            nodes.append(node)
            scores.append(score)
            ids.append(memory["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=scores, ids=ids)

    # ------------------------------------------------------------------
    # VectorStore interface — asynchronous
    # ------------------------------------------------------------------

    async def async_add(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> List[str]:
        """Async variant of :meth:`add`."""
        ids: List[str] = []
        for node in nodes:
            payload = self._build_store_payload(node)
            try:
                resp = await self._async_client.post(
                    "/v1/memory/store", json=payload
                )
                resp.raise_for_status()
                memory_id: str = resp.json()["memory"]["id"]
                ids.append(memory_id)
            except Exception as exc:
                logger.error(
                    "Dakera async store failed for node %s: %s", node.node_id, exc
                )
                raise
        return ids

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Async variant of :meth:`delete`."""
        memory_ids: Optional[List[str]] = delete_kwargs.get("memory_ids")

        if memory_ids is None:
            try:
                resp = await self._async_client.post(
                    "/v1/memory/search",
                    json={
                        "agent_id": self.agent_id,
                        "query": f"node_id={ref_doc_id}",
                        "top_k": 100,
                    },
                )
                resp.raise_for_status()
                hits = resp.json().get("memories", [])
                memory_ids = [
                    h["memory"]["id"]
                    for h in hits
                    if any(
                        t == f"node_id={ref_doc_id}"
                        for t in (h["memory"].get("tags") or [])
                    )
                ]
            except Exception as exc:
                logger.error(
                    "Dakera async search-before-delete failed for %s: %s",
                    ref_doc_id,
                    exc,
                )
                raise

        if not memory_ids:
            return

        try:
            resp = await self._async_client.post(
                "/v1/memory/forget",
                json={"agent_id": self.agent_id, "memory_ids": memory_ids},
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.error("Dakera async forget failed for %s: %s", memory_ids, exc)
            raise

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Async variant of :meth:`query`."""
        if not query.query_str:
            raise ValueError(
                "DakeraVectorStore requires query.query_str (text). "
                "Pre-computed embeddings are not supported — Dakera embeds server-side."
            )

        top_k = query.similarity_top_k or self.top_k
        payload: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "query": query.query_str,
            "top_k": top_k,
        }
        if self.session_id:
            payload["session_id"] = self.session_id

        try:
            resp = await self._async_client.post("/v1/memory/search", json=payload)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("Dakera async search failed: %s", exc)
            raise

        hits = resp.json().get("memories", [])
        nodes: List[BaseNode] = []
        scores: List[float] = []
        ids: List[str] = []

        for hit in hits:
            memory = hit["memory"]
            score: float = float(hit.get("score", 0.0))
            node = self._memory_to_node(memory)
            nodes.append(node)
            scores.append(score)
            ids.append(memory["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=scores, ids=ids)
