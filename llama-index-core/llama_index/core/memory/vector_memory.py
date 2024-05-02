"""Vector memory.

Memory backed by a vector database.

"""

import json
from typing import Any, Callable, Dict, List, Optional

from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field, root_validator
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.types import DEFAULT_CHAT_STORE_KEY, BaseMemory
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
from llama_index.core.utils import get_tokenizer
from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model

from llama_index.core.indices.vector_store import VectorStoreIndex

DEFAULT_TOKEN_LIMIT_RATIO = 0.75
DEFAULT_TOKEN_LIMIT = 3000



DEFAULT_SYSTEM_MESSAGE = "This is a set of relevant messages retrieved from longer-term history: "


class VectorMemory(BaseMemory):
    """Memory backed by a vector index."""

    vector_index: VectorStoreIndex
    retriever_kwargs: Dict[str, Any] = Field(default_factory=dict)

    system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE

    # whether to condense all memory into a single message
    return_single_message: bool = False

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "VectorMemory"

    @classmethod
    def from_defaults(
        cls,
        vector_store: Optional[VectorStore] = None,
        embed_model: Optional[EmbedType] = None,
        index_kwargs: Optional[Dict] = None,
        retriever_kwargs: Optional[Dict] = None,
    ) -> "VectorMemory":
        """Create vector memory.

        Args:
            vector_store (Optional[VectorStore]): vector store
            embed_model (Optional[EmbedType]): embedding model
            index_kwargs (Optional[Dict]): kwargs for initializing the index
            retriever_kwargs (Optional[Dict]): kwargs for initializing the retriever
        
        """
        index_kwargs = index_kwargs or {}
        retriever_kwargs = retriever_kwargs or {}
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model, **index_kwargs)
        return cls(vector_index=index, retriever_kwargs=retriever_kwargs)

    def get(self, input: Optional[str] = None, initial_token_count: int = 0, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history."""
        if input is None:
            raise ValueError("Input must be provided to get chat history.")

        # retrieve from index
        retriever = self.vector_index.get_retriever(**self.retriever_kwargs)
        nodes = retriever.retrieve(input or "")
        messages = [ChatMessage.from_node(node) for node in nodes]

        # add system message
        if self.system_message:
            messages = [
                ChatMessage.from_str(self.system_message, role=MessageRole.SYSTEM)] + messages


        if self.return_single_message:
            # condense all messages into a single message
            messages = [
                ChatMessage.from_str(" ".join([m.content for m in messages]), role=MessageRole.USER)
            ]

        return messages

    def get_all(self) -> List[ChatMessage]:
        """Get all chat history."""
        # TODO: 
        raise ValueError(
            "Vector memory does not support get_all method, can only retrieve based on input."
        )

    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        # insert into index
        # ensure everything is serialized

        self.vector_index.insert_nodes([message.to_node()])

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history."""
        # TODO: implementation for later
        self.reset()
        for message in messages:
            self.put(message)

    def reset(self) -> None:
        """Reset chat history."""
        raise NotImplementedError(
            "As of right now, our vector store abstractions do not support "
            "dropping an entire collection. If you are using this vector memory "
            "module, please use the relevant vector store SDK to drop "
            "the collection whenever `memory.reset()` or `agent.reset()` is called. "
        )