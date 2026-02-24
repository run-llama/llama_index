"""OpenSearch chat store."""

from typing import Any, Dict, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore

IMPORT_ERROR_MSG = (
    "Could not import opensearch-py. "
    "Please install it with `pip install opensearch-py`."
)

DEFAULT_INDEX_NAME = "llama_index_chat_store"

# OpenSearch index mapping for chat messages
CHAT_STORE_MAPPING: Dict[str, Any] = {
    "mappings": {
        "properties": {
            "session_id": {"type": "keyword"},
            "index": {"type": "integer"},
            "message": {"type": "keyword", "index": False, "doc_values": False},
        }
    },
}


def _message_to_str(message: ChatMessage) -> str:
    """Serialize a ChatMessage to a JSON string for storage."""
    return message.model_dump_json()


def _str_to_message(s: str) -> ChatMessage:
    """Deserialize a JSON string from storage to a ChatMessage."""
    return ChatMessage.model_validate_json(s)


def _get_opensearch_client(opensearch_url: str, **kwargs: Any) -> "OpenSearch":
    """Get an OpenSearch client from URL."""
    try:
        from opensearchpy import OpenSearch
    except ImportError:
        raise ImportError(IMPORT_ERROR_MSG)

    return OpenSearch(opensearch_url, **kwargs)


def _get_async_opensearch_client(
    opensearch_url: str, **kwargs: Any
) -> "AsyncOpenSearch":
    """Get an async OpenSearch client from URL."""
    try:
        from opensearchpy import AsyncOpenSearch
    except ImportError:
        raise ImportError(IMPORT_ERROR_MSG)

    return AsyncOpenSearch(opensearch_url, **kwargs)


class OpensearchChatStore(BaseChatStore):
    """
    OpenSearch chat store.

    Stores chat messages as individual documents in an OpenSearch index,
    keyed by session_id with an integer index for ordering.

    Args:
        opensearch_url: OpenSearch endpoint URL.
        index: Name of the OpenSearch index to store messages in.
        os_client: Optional pre-configured OpenSearch client.
        os_async_client: Optional pre-configured async OpenSearch client.
        **kwargs: Additional arguments passed to the OpenSearch client.

    """

    opensearch_url: str = Field(
        default="https://localhost:9200",
        description="OpenSearch URL.",
    )
    index: str = Field(
        default=DEFAULT_INDEX_NAME,
        description="OpenSearch index name for chat messages.",
    )

    _os_client: Any = PrivateAttr()
    _os_async_client: Any = PrivateAttr()

    def __init__(
        self,
        opensearch_url: str = "https://localhost:9200",
        index: str = DEFAULT_INDEX_NAME,
        os_client: Optional[Any] = None,
        os_async_client: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpensearchChatStore."""
        super().__init__(opensearch_url=opensearch_url, index=index)

        self._os_client = os_client or _get_opensearch_client(opensearch_url, **kwargs)
        self._os_async_client = os_async_client or _get_async_opensearch_client(
            opensearch_url, **kwargs
        )

        self._ensure_index_exists()

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "OpensearchChatStore"

    def _ensure_index_exists(self) -> None:
        """Create the index if it does not already exist."""
        if not self._os_client.indices.exists(index=self.index):
            self._os_client.indices.create(index=self.index, body=CHAT_STORE_MAPPING)

    async def _aensure_index_exists(self) -> None:
        """Async: create the index if it does not already exist."""
        exists = await self._os_async_client.indices.exists(index=self.index)
        if not exists:
            await self._os_async_client.indices.create(
                index=self.index, body=CHAT_STORE_MAPPING
            )

    # ---- helpers ----

    def _search(self, query: Dict[str, Any], size: int = 10000) -> List[Dict]:
        """Run a search and return the list of hits."""
        resp = self._os_client.search(index=self.index, body=query, size=size)
        return resp["hits"]["hits"]

    async def _asearch(self, query: Dict[str, Any], size: int = 10000) -> List[Dict]:
        """Async: run a search and return the list of hits."""
        resp = await self._os_async_client.search(
            index=self.index, body=query, size=size
        )
        return resp["hits"]["hits"]

    def _delete_by_query(self, query: Dict[str, Any]) -> None:
        """Delete documents matching a query."""
        self._os_client.delete_by_query(
            index=self.index,
            body=query,
            refresh=True,
        )

    async def _adelete_by_query(self, query: Dict[str, Any]) -> None:
        """Async: delete documents matching a query."""
        await self._os_async_client.delete_by_query(
            index=self.index,
            body=query,
            refresh=True,
        )

    def _session_query(self, key: str) -> Dict[str, Any]:
        """Build a query to match all documents for a session."""
        return {"query": {"term": {"session_id": key}}}

    def _session_sorted_query(self, key: str, order: str = "asc") -> Dict[str, Any]:
        """Build a query for a session, sorted by index."""
        return {
            "query": {"term": {"session_id": key}},
            "sort": [{"index": {"order": order}}],
        }

    def _find_by_index_query(self, key: str, idx: int) -> Dict[str, Any]:
        """Build a query to match a single document by session + index."""
        return {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"session_id": key}},
                        {"term": {"index": idx}},
                    ]
                }
            }
        }

    def _shift_query(self, key: str, from_idx: int) -> Dict[str, Any]:
        """Build a query to find documents at or after a given index (desc)."""
        return {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"session_id": key}},
                        {"range": {"index": {"gte": from_idx}}},
                    ]
                }
            },
            "sort": [{"index": {"order": "desc"}}],
        }

    def _get_next_index(self, key: str) -> int:
        """Get the next available index for a session."""
        hits = self._search(self._session_sorted_query(key, order="desc"), size=1)
        if not hits:
            return 0
        return int(hits[0]["_source"]["index"]) + 1

    async def _aget_next_index(self, key: str) -> int:
        """Async: get the next available index for a session."""
        hits = await self._asearch(
            self._session_sorted_query(key, order="desc"), size=1
        )
        if not hits:
            return 0
        return int(hits[0]["_source"]["index"]) + 1

    def _index_doc(self, key: str, idx: int, message: ChatMessage) -> None:
        """Index a single message document."""
        self._os_client.index(
            index=self.index,
            body={
                "session_id": key,
                "index": idx,
                "message": _message_to_str(message),
            },
            refresh=True,
        )

    async def _aindex_doc(self, key: str, idx: int, message: ChatMessage) -> None:
        """Async: index a single message document."""
        await self._os_async_client.index(
            index=self.index,
            body={
                "session_id": key,
                "index": idx,
                "message": _message_to_str(message),
            },
            refresh=True,
        )

    def _reindex_session(self, key: str) -> None:
        """Re-number all documents in a session so indices are contiguous."""
        hits = self._search(self._session_sorted_query(key))
        # Delete all existing documents for this session
        self._delete_by_query(self._session_query(key))
        # Re-insert with corrected indices
        for new_idx, hit in enumerate(hits):
            msg = _str_to_message(hit["_source"]["message"])
            self._index_doc(key, new_idx, msg)

    async def _areindex_session(self, key: str) -> None:
        """Async: re-number all documents in a session so indices are contiguous."""
        hits = await self._asearch(self._session_sorted_query(key))
        # Delete all existing documents for this session
        await self._adelete_by_query(self._session_query(key))
        # Re-insert with corrected indices
        for new_idx, hit in enumerate(hits):
            msg = _str_to_message(hit["_source"]["message"])
            await self._aindex_doc(key, new_idx, msg)

    # ---- BaseChatStore interface ----

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Set messages for a key, replacing any existing messages."""
        # Delete existing messages for this session
        self._delete_by_query(self._session_query(key))

        # Insert new messages
        for idx, message in enumerate(messages):
            self._index_doc(key, idx, message)

    async def aset_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Async: set messages for a key, replacing any existing messages."""
        await self._adelete_by_query(self._session_query(key))

        for idx, message in enumerate(messages):
            await self._aindex_doc(key, idx, message)

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a key, ordered by index."""
        hits = self._search(self._session_sorted_query(key))
        return [_str_to_message(hit["_source"]["message"]) for hit in hits]

    async def aget_messages(self, key: str) -> List[ChatMessage]:
        """Async: get messages for a key, ordered by index."""
        hits = await self._asearch(self._session_sorted_query(key))
        return [_str_to_message(hit["_source"]["message"]) for hit in hits]

    def add_message(
        self, key: str, message: ChatMessage, idx: Optional[int] = None
    ) -> None:
        """
        Add a message for a key.

        If idx is None, appends to the end. Otherwise inserts at the given
        position and shifts subsequent messages.
        """
        if idx is None:
            idx = self._get_next_index(key)
            self._index_doc(key, idx, message)
        else:
            # Shift existing messages at >= idx up by one (reverse to avoid collisions)
            for hit in self._search(self._shift_query(key, idx)):
                self._os_client.update(
                    index=self.index,
                    id=hit["_id"],
                    body={"doc": {"index": hit["_source"]["index"] + 1}},
                    refresh=True,
                )
            self._index_doc(key, idx, message)

    async def async_add_message(
        self, key: str, message: ChatMessage, idx: Optional[int] = None
    ) -> None:
        """Async: add a message for a key."""
        if idx is None:
            idx = await self._aget_next_index(key)
            await self._aindex_doc(key, idx, message)
        else:
            for hit in await self._asearch(self._shift_query(key, idx)):
                await self._os_async_client.update(
                    index=self.index,
                    id=hit["_id"],
                    body={"doc": {"index": hit["_source"]["index"] + 1}},
                    refresh=True,
                )
            await self._aindex_doc(key, idx, message)

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Delete all messages for a key. Returns the deleted messages."""
        messages = self.get_messages(key)
        self._delete_by_query(self._session_query(key))
        return messages if messages else None

    async def adelete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Async: delete all messages for a key."""
        messages = await self.aget_messages(key)
        await self._adelete_by_query(self._session_query(key))
        return messages if messages else None

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """
        Delete a specific message by index for a key.

        After deletion, remaining messages are re-indexed to stay contiguous.
        """
        hits = self._search(self._find_by_index_query(key, idx), size=1)
        if not hits:
            return None

        deleted_message = _str_to_message(hits[0]["_source"]["message"])
        self._os_client.delete(index=self.index, id=hits[0]["_id"], refresh=True)
        self._reindex_session(key)
        return deleted_message

    async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Async: delete a specific message by index for a key."""
        hits = await self._asearch(self._find_by_index_query(key, idx), size=1)
        if not hits:
            return None

        deleted_message = _str_to_message(hits[0]["_source"]["message"])
        await self._os_async_client.delete(
            index=self.index, id=hits[0]["_id"], refresh=True
        )
        await self._areindex_session(key)
        return deleted_message

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete the last message for a key."""
        hits = self._search(self._session_sorted_query(key, order="desc"), size=1)
        if not hits:
            return None

        last_message = _str_to_message(hits[0]["_source"]["message"])
        self._os_client.delete(index=self.index, id=hits[0]["_id"], refresh=True)
        return last_message

    async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Async: delete the last message for a key."""
        hits = await self._asearch(
            self._session_sorted_query(key, order="desc"), size=1
        )
        if not hits:
            return None

        last_message = _str_to_message(hits[0]["_source"]["message"])
        await self._os_async_client.delete(
            index=self.index, id=hits[0]["_id"], refresh=True
        )
        return last_message

    def get_keys(self) -> List[str]:
        """Get all unique session keys."""
        query = {
            "size": 0,
            "aggs": {
                "unique_sessions": {"terms": {"field": "session_id", "size": 10000}}
            },
        }
        resp = self._os_client.search(index=self.index, body=query)
        buckets = resp["aggregations"]["unique_sessions"]["buckets"]
        return [bucket["key"] for bucket in buckets]

    async def aget_keys(self) -> List[str]:
        """Async: get all unique session keys."""
        query = {
            "size": 0,
            "aggs": {
                "unique_sessions": {"terms": {"field": "session_id", "size": 10000}}
            },
        }
        resp = await self._os_async_client.search(index=self.index, body=query)
        buckets = resp["aggregations"]["unique_sessions"]["buckets"]
        return [bucket["key"] for bucket in buckets]
