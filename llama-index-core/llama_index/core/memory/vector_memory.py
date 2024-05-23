"""Vector memory.

Memory backed by a vector database.

"""

import uuid
from typing import Any, Dict, List, Optional
from llama_index.core.bridge.pydantic import validator

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field
from llama_index.core.memory.types import DEFAULT_CHAT_STORE_KEY, BaseMemory
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
from llama_index.core.embeddings.utils import EmbedType


DEFAULT_TOKEN_LIMIT_RATIO = 0.75
DEFAULT_TOKEN_LIMIT = 3000


DEFAULT_SYSTEM_MESSAGE = (
    "This is a set of relevant messages retrieved from longer-term history: "
)


def _stringify_obj(d: Any):
    """Utility function to convert all keys in a dictionary to strings."""
    if isinstance(d, list):
        return [_stringify_obj(v) for v in d]
    elif isinstance(d, Dict):
        return {str(k): _stringify_obj(v) for k, v in d.items()}
    else:
        return str(d)


def _stringify_chat_message(msg: ChatMessage) -> Dict:
    """Utility function to convert chatmessage to serializable dict."""
    msg_dict = msg.dict()
    msg_dict["additional_kwargs"] = _stringify_obj(msg_dict["additional_kwargs"])
    return msg_dict


CUR_USER_MSG_KEY = "cur_user_msg"


class VectorMemory(BaseMemory):
    """Memory backed by a vector index."""

    vector_index: Any
    retriever_kwargs: Dict[str, Any] = Field(default_factory=dict)

    system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE

    # whether to condense all memory into a single message
    return_single_message: bool = True

    # Whether to combine a user message with all subsequent messages
    # until the next user message into a single message
    # This is on by default, ensuring that we always fetch contiguous blocks of user/response pairs.
    # Turning this off may lead to errors in the function calling API of the LLM.
    # If this is on, then any message that's not a user message will be combined with the last user message
    # in the vector store.
    batch_by_user_message: bool = True

    chat_store: BaseChatStore = Field(default_factory=SimpleChatStore)
    # NOTE/TODO: we need this to store id's for the messages
    # This is not needed once vector stores implement delete_all capabilities
    chat_store_key: str = Field(default=DEFAULT_CHAT_STORE_KEY)
    # NOTE: this is to store the current user message batch (if `batch_by_user_message` is True)
    # allows us to keep track of the current user message batch
    # so we can delete it when we commit a new node
    cur_user_msg_key: str = Field(default=CUR_USER_MSG_KEY)

    @validator("vector_index")
    def validate_vector_index(cls, value: Any) -> Any:
        """Validate vector index."""
        # NOTE: we can't import VectorStoreIndex directly due to circular imports,
        # which is why the type is Any
        from llama_index.core.indices.vector_store import VectorStoreIndex

        if not isinstance(value, VectorStoreIndex):
            raise ValueError(
                f"Expected 'vector_index' to be an instance of VectorStoreIndex, got {type(value)}"
            )
        return value

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
        from llama_index.core.indices.vector_store import VectorStoreIndex

        index_kwargs = index_kwargs or {}
        retriever_kwargs = retriever_kwargs or {}

        if vector_store is None:
            # initialize a blank in-memory vector store
            # NOTE: can't easily do that from `from_vector_store` at the moment.
            index = VectorStoreIndex.from_documents(
                [], embed_model=embed_model, **index_kwargs
            )
        else:
            index = VectorStoreIndex.from_vector_store(
                vector_store, embed_model=embed_model, **index_kwargs
            )
        return cls(vector_index=index, retriever_kwargs=retriever_kwargs)

    def get(
        self, input: Optional[str] = None, initial_token_count: int = 0, **kwargs: Any
    ) -> List[ChatMessage]:
        """Get chat history."""
        if input is None:
            raise ValueError("Input must be provided to get chat history.")

        # retrieve from index
        retriever = self.vector_index.as_retriever(**self.retriever_kwargs)
        nodes = retriever.retrieve(input or "")

        # retrieve underlying messages
        return [
            ChatMessage.parse_obj(sub_dict)
            for node in nodes
            for sub_dict in node.metadata["sub_dicts"]
        ]

    def get_all(self) -> List[ChatMessage]:
        """Get all chat history."""
        # TODO: while we could implement get_all, would be hacky through metadata filtering
        # since vector stores don't easily support get()
        raise ValueError(
            "Vector memory does not support get_all method, can only retrieve based on input."
        )

    def _commit_node(self, override_last: bool = False) -> None:
        """Commit new node to vector store."""
        # commit to vector store
        node_id = str(uuid.uuid4())
        # create subnodes for each message
        sub_dicts = []
        for msg in self.chat_store.get_messages(self.cur_user_msg_key):
            sub_dicts.append(_stringify_chat_message(msg))

        if not sub_dicts:
            return

        # now create a "super" node that contains all subnodes as metadata
        # this metadata is excluded from embedding and LLM synthesis
        # the concatenated text is put into the super node text field
        super_node = TextNode(
            text=" ".join(
                [str(sub_dicts[i]["content"]) for i in range(len(sub_dicts))]
            ),
            id_=node_id,
            metadata={"sub_dicts": sub_dicts},
            excluded_embed_metadata_keys=["sub_dicts"],
            excluded_llm_metadata_keys=["sub_dicts"],
        )

        if override_last:
            # delete the last node
            # This is needed since we're updating the last node in the vector
            # index as its being updated. When a new user-message batch starts
            # we already will have the last user message group committed to the
            # vector store index and so we don't need to override_last (i.e. see
            # logic in self.put().)
            last_node_id = self.chat_store.delete_last_message(
                self.chat_store_key
            ).content
            self.vector_index.delete_nodes([last_node_id])

        self.vector_index.insert_nodes([super_node])
        self.chat_store.add_message(self.chat_store_key, ChatMessage(content=node_id))

    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        if not self.batch_by_user_message or message.role == MessageRole.USER:
            # if not batching by user message, commit to vector store immediately after adding
            self.chat_store.delete_messages(self.cur_user_msg_key)
            self.chat_store.add_message(self.cur_user_msg_key, message)
            self._commit_node()
        else:
            # if not user message, add to holding queue i.e. the chat_store
            self.chat_store.add_message(self.cur_user_msg_key, message)
            self._commit_node(override_last=True)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history."""
        self.reset()
        for message in messages:
            self.put(message)

    def reset(self) -> None:
        """Reset chat history."""
        node_id_msgs = self.chat_store.get_messages(self.chat_store_key)
        node_ids = [msg.content for msg in node_id_msgs]
        self.vector_index.delete_nodes(node_ids)

        # delete from chat history
        self.chat_store.delete_messages(self.chat_store_key)


VectorMemory.update_forward_refs()
