"""Vector memory.

Memory backed by a vector database.

"""

import json
import uuid
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from llama_index.core.bridge.pydantic import validator

from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.schema import TextNode, BaseNode, RelatedNodeInfo, NodeRelationship
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field, root_validator
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.types import DEFAULT_CHAT_STORE_KEY, BaseMemory
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
from llama_index.core.utils import get_tokenizer
from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model

if TYPE_CHECKING:
    from llama_index.core.indices.vector_store import VectorStoreIndex

DEFAULT_TOKEN_LIMIT_RATIO = 0.75
DEFAULT_TOKEN_LIMIT = 3000



DEFAULT_SYSTEM_MESSAGE = "This is a set of relevant messages retrieved from longer-term history: "

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
    
    # if not isinstance(d, Dict):
    #     return str(d)
    # return {str(k): _stringify_dict(v) for k, v in d.items()}

# def to_node(message: ChatMessage, node_id: Optional[str] = None, **node_kwargs: Any) -> BaseNode:
#     """Convert to node.

#     Non-stringable objects will get stringified.
    
#     """
#     from llama_index.core.schema import TextNode

#     add_kwargs_dict = _stringify_dict(message.additional_kwargs)

#     return TextNode(
#         text=str(message.content),
#         id_=node_id,
#         metadata={"role": message.role.value, **add_kwargs_dict},
#     )

# def from_node(node: "BaseNode") -> "ChatMessage":
#     """Create from node."""
#     return ChatMessage(
#         role=MessageRole(node.metadata.get("role", MessageRole.USER.value)),
#         content=node.text,
#         additional_kwargs={k: v for k, v in node.metadata.items() if k != "role"},
#     )


CUR_USER_MSG_KEY = "cur_user_msg"


class VectorMemory(BaseMemory):
    """Memory backed by a vector index."""

    vector_index: Any
    retriever_kwargs: Dict[str, Any] = Field(default_factory=dict)

    system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE

    # whether to condense all memory into a single message
    return_single_message: bool = False
    
    # Whether to combine a user message with all subsequent messages 
    # until the next user message into a single message
    # This is on by default, ensuring that we always fetch contiguous blocks of user/response pairs.
    # Turning this off may lead to errors in the function calling API of the LLM.
    # If this is on, we don't commit to vector store until all the messages are present.
    # `get()` will also return the latest user message 
    # that is "incomplete" (hasn't been committed to the vector store)
    batch_by_user_message: bool = True

    # NOTE/TODO: we need this to store id's for the messages
    # This is not needed once vector stores implement delete_all capabilities
    chat_store: BaseChatStore = Field(default_factory=SimpleChatStore)
    chat_store_key: str = Field(default=DEFAULT_CHAT_STORE_KEY)
    cur_user_msg_key: str = Field(default=CUR_USER_MSG_KEY)

    @validator('vector_index')
    def validate_vector_index(cls, value: Any) -> Any:
        """Validate vector index."""
        # NOTE: we can't import VectorStoreIndex directly due to circular imports,
        # which is why the type is Any
        from llama_index.core.indices.vector_store import VectorStoreIndex
        if not isinstance(value, VectorStoreIndex):
            raise ValueError(f"Expected 'vector_index' to be an instance of VectorStoreIndex, got {type(value)}")
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
            index = VectorStoreIndex.from_documents([], embed_model=embed_model, **index_kwargs)
        else:
            index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model, **index_kwargs)
        return cls(vector_index=index, retriever_kwargs=retriever_kwargs)

    def get(self, input: Optional[str] = None, initial_token_count: int = 0, **kwargs: Any) -> List[ChatMessage]:
        """Get chat history."""
        if input is None:
            raise ValueError("Input must be provided to get chat history.")

        # retrieve from index
        retriever = self.vector_index.as_retriever(**self.retriever_kwargs)
        nodes = retriever.retrieve(input or "")
        # messages = [from_node(node) for node in nodes]

        # retrieve underlying messages
        # messages = [from_node(sub_node) for node in nodes for sub_node in node.metadata["sub_nodes"]]
        messages = [
            ChatMessage.parse_obj(sub_dict) 
            for node in nodes for sub_dict in node.metadata["sub_dicts"]
        ]

        # add system message
        if self.system_message:
            messages = [
                ChatMessage.from_str(self.system_message, role=MessageRole.SYSTEM)] + messages

        # # if batching by user message, return the latest user message that is "incomplete"
        # if self.batch_by_user_message:
        #     # get the latest user message
        #     user_msg = self.chat_store.get_messages(self.cur_user_msg_key)
        #     if user_msg:
        #         messages = messages + user_msg

        if self.return_single_message:
            # condense all messages into a single message
            messages = [
                ChatMessage.from_str(" ".join([m.content for m in messages]), role=MessageRole.USER)
            ]

        print(f"Memory returning messages: {messages}")

        return messages

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
            # sub_nodes.append(to_node(msg))
            # sub_dicts.append(_stringify_dict(msg.dict()))
            sub_dicts.append(_stringify_chat_message(msg))

        if not sub_dicts:
            return

        # now create a "super" node that contains all subnodes as metadata
        # this metadata is excluded from embedding and LLM synthesis
        # the concatenated text is put into the super node text field
        super_node = TextNode(
            text=" ".join([str(sub_dicts[i]["content"]) for i in range(len(sub_dicts))]),
            id_=node_id,
            metadata={"sub_dicts": sub_dicts},
            excluded_embed_metadata_keys=["sub_dicts"],
            excluded_llm_metadata_keys=["sub_dicts"],
        )
        # super_node = TextNode(
        #     text=" ".join([node.get_content() for node in sub_nodes]),
        #     id_=node_id,
        #     metadata={"sub_nodes": [node.dict() for node in sub_nodes]},
        #     excluded_embed_metadata_keys=["sub_nodes"],
        #     excluded_llm_metadata_keys=["sub_nodes"],
        # )
        # HACK: this is a hack to add the source relationship as itself, to make deletion work.
        super_node.relationships[NodeRelationship.SOURCE] = super_node.as_related_node_info()

        if override_last:
            # delete the last node
            # last_node_id = self.chat_store.get_messages(self.chat_store_key)[-1].content
            last_node_id = self.chat_store.delete_last_message(self.chat_store_key).content
            self.vector_index.delete_ref_doc(last_node_id)

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
            # if not user message, add to holding queue
            self.chat_store.add_message(self.cur_user_msg_key, message)
            self._commit_node(override_last=True)
        
        # insert into index
        # ensure everything is serialized

        # # assign node id
        # node_id = str(uuid.uuid4())
        # message_node = to_node(message, node_id=node_id)
        # # HACK: this is a hack to add the source relationship as itself, to make deletion work.
        # message_node.relationships[NodeRelationship.SOURCE] = message_node.as_related_node_info()

        # self.chat_store.add_message(self.chat_store_key, ChatMessage(content=node_id))
        # self.vector_index.insert_nodes([message_node])

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

        # delete from chat history
        self.chat_store.delete_messages(self.chat_store_key)


VectorMemory.update_forward_refs()