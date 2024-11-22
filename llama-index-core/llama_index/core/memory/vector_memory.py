"""Vector memory.

Memory backed by a vector database.

"""

import uuid
from typing import Any, Dict, List, Optional, Union

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field, field_validator
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.memory.types import BaseMemory
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore


def _stringify_obj(d: Any) -> Union[str, list, dict]:
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
    msg_dict["content"] = msg.content
    return msg_dict


def _get_starter_node_for_new_batch() -> TextNode:
    """Generates a new starter node for a new batch or group of messages."""
    return TextNode(
        id_=str(uuid.uuid4()),
        text="",
        metadata={"sub_dicts": []},
        excluded_embed_metadata_keys=["sub_dicts"],
        excluded_llm_metadata_keys=["sub_dicts"],
    )


class VectorMemory(BaseMemory):
    """Memory backed by a vector index.

    NOTE: This class requires the `delete_nodes` method to be implemented
    by the vector store underlying the vector index. At time of writing (May 2024),
    Chroma, Qdrant and SimpleVectorStore all support delete_nodes.
    """

    vector_index: Any
    retriever_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # Whether to combine a user message with all subsequent messages
    # until the next user message into a single message
    # This is on by default, ensuring that we always fetch contiguous blocks of user/response pairs.
    # Turning this off may lead to errors in the function calling API of the LLM.
    # If this is on, then any message that's not a user message will be combined with the last user message
    # in the vector store.
    batch_by_user_message: bool = True

    cur_batch_textnode: TextNode = Field(
        default_factory=_get_starter_node_for_new_batch,
        description="The super node for the current active user-message batch.",
    )

    @field_validator("vector_index")
    @classmethod
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
        vector_store: Optional[BasePydanticVectorStore] = None,
        embed_model: Optional[EmbedType] = None,
        index_kwargs: Optional[Dict] = None,
        retriever_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> "VectorMemory":
        """Create vector memory.

        Args:
            vector_store (Optional[BasePydanticVectorStore]): vector store (note: delete_nodes must
                be implemented. At time of writing (May 2024), Chroma, Qdrant and
                SimpleVectorStore all support delete_nodes.
            embed_model (Optional[EmbedType]): embedding model
            index_kwargs (Optional[Dict]): kwargs for initializing the index
            retriever_kwargs (Optional[Dict]): kwargs for initializing the retriever

        """
        from llama_index.core.indices.vector_store import VectorStoreIndex

        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")

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
            return []

        # retrieve from index
        retriever = self.vector_index.as_retriever(**self.retriever_kwargs)
        nodes = retriever.retrieve(input or "")

        # retrieve underlying messages
        return [
            ChatMessage.model_validate(sub_dict)
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
        if self.cur_batch_textnode.text == "":
            return

        if override_last:
            # delete the last node
            # This is needed since we're updating the last node in the vector
            # index as its being updated. When a new user-message batch starts
            # we already will have the last user message group committed to the
            # vector store index and so we don't need to override_last (i.e. see
            # logic in self.put().)
            self.vector_index.delete_nodes([self.cur_batch_textnode.id_])

        self.vector_index.insert_nodes([self.cur_batch_textnode])

    def put(self, message: ChatMessage) -> None:
        """Put chat history."""
        if not self.batch_by_user_message or message.role in [
            MessageRole.USER,
            MessageRole.SYSTEM,
        ]:
            # if not batching by user message, commit to vector store immediately after adding
            self.cur_batch_textnode = _get_starter_node_for_new_batch()

        # update current batch textnode
        sub_dict = _stringify_chat_message(message)
        if self.cur_batch_textnode.text == "":
            self.cur_batch_textnode.text += sub_dict["content"] or ""
        else:
            self.cur_batch_textnode.text += " " + (sub_dict["content"] or "")
        self.cur_batch_textnode.metadata["sub_dicts"].append(sub_dict)
        self._commit_node(override_last=True)

    def set(self, messages: List[ChatMessage]) -> None:
        """Set chat history."""
        self.reset()
        for message in messages:
            self.put(message)

    def reset(self) -> None:
        """Reset chat history."""
        self.vector_index.vector_store.clear()


VectorMemory.model_rebuild()
