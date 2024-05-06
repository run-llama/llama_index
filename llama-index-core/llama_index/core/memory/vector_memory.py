"""Vector memory.

Memory backed by a vector database.

"""

import json
import uuid
from typing import Any, Callable, Dict, List, Optional

from llama_index.core.schema import TextNode, BaseNode, RelatedNodeInfo, NodeRelationship
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

def to_node(message: ChatMessage, node_id: Optional[str] = None, **node_kwargs: Any) -> BaseNode:
    """Convert to node."""
    from llama_index.core.schema import BaseNode

    return BaseNode(
        text=message.content,
        id_=node_id,
        metadata={"role": message.role.value, **message.additional_kwargs},
    )

def from_node(node: "BaseNode") -> "ChatMessage":
    """Create from node."""
    return ChatMessage(
        role=MessageRole(node.metadata.get("role", MessageRole.USER.value)),
        content=node.text,
        additional_kwargs={k: v for k, v in node.metadata.items() if k != "role"},
    )


class VectorMemory(BaseMemory):
    """Memory backed by a vector index."""

    vector_index: VectorStoreIndex
    retriever_kwargs: Dict[str, Any] = Field(default_factory=dict)

    system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE

    # whether to condense all memory into a single message
    return_single_message: bool = False

    # NOTE/TODO: we need this to store id's for the messages
    # This is not needed once vector stores implement delete_all capabilities
    chat_store: BaseChatStore = Field(default_factory=SimpleChatStore)
    chat_store_key: str = Field(default=DEFAULT_CHAT_STORE_KEY)

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
        # TODO: while we could implement get_all, would be hacky through metadata filtering
        # since vector stores don't easily support get()
        raise ValueError(
            "Vector memory does not support get_all method, can only retrieve based on input."
        )

    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        # insert into index
        # ensure everything is serialized

        # assign node id
        node_id = str(uuid.uuid4())
        message_node = to_node(message, node_id=node_id, )
        # HACK: this is a hack to add the source relationship as itself, to make deletion work.
        message_node.relationships[NodeRelationship.SOURCE] = message_node.as_related_node_info()

        self.chat_store.add_message(self.chat_store_key, ChatMessage(content=node_id))
        self.vector_index.insert_nodes([message_node])

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history."""
        self.reset()
        for message in messages:
            self.put(message)

    def reset(self) -> None:
        """Reset chat history."""
        node_id_msgs = self.chat_store.get_messages(self.chat_store_key)
        node_ids = [msg.content for msg in node_id_msgs]
        [self.vector_index.delete_ref_doc(node_id) for node_id in node_ids]
