import json
import os
from typing import Dict, List, Optional

import fsspec

from llama_index.bridge.pydantic import Field
from llama_index.llms import ChatMessage
from llama_index.storage.chat_store.base import BaseChatStore


class SimpleChatStore(BaseChatStore):
    """Simple chat store."""

    store: Dict[str, List[ChatMessage]] = Field(default_factory=dict)

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "SimpleChatStore"

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Set messages for a key."""
        self.store[key] = messages

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a key."""
        return self.store.get(key, [])

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message for a key."""
        self.store.setdefault(key, []).append(message)

    def delete_messages(self, key: str) -> None:
        """Delete messages for a key."""
        self.store.pop(key, None)

    def undo_last_message(self, key: str) -> Optional[ChatMessage]:
        """Undo last message for a key."""
        return self.store.get(key, []).pop()

    def get_keys(self) -> List[str]:
        """Get all keys."""
        return list(self.store.keys())

    def persist(
        self,
        persist_path: str = "chat_store.json",
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the docstore to a file."""
        fs = fs or fsspec.filesystem("file")
        dirpath = os.path.dirname(persist_path)
        if not fs.exists(dirpath):
            fs.makedirs(dirpath)

        with fs.open(persist_path, "w") as f:
            f.write(json.dumps(self.store))

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str = "chat_store.json",
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "SimpleChatStore":
        """Create a SimpleChatStore from a persist path."""
        fs = fs or fsspec.filesystem("file")
        if not fs.exists(persist_path):
            return cls()
        with fs.open(persist_path, "r") as f:
            store = json.load(f)
        return cls(store=store)
