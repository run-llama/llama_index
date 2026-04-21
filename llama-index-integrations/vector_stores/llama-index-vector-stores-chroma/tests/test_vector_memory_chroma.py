import json
import uuid
from typing import Any, Dict

import chromadb

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import VectorMemory
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore


def _node_to_chroma_metadata(node: TextNode) -> Dict[str, Any]:
    """
    Create Chroma-safe metadata while preserving full node in _node_content.

    Chroma's metadata values must be scalars, but we still want node reconstruction
    (including nested metadata like `sub_dicts`) via `_node_content`.
    """
    node_dict = node.model_dump(mode="json")
    node_dict["text"] = ""
    node_dict["embedding"] = None

    return {
        "_node_content": json.dumps(node_dict, ensure_ascii=False),
        "_node_type": node.class_name(),
        "document_id": node.ref_doc_id or "None",
        "doc_id": node.ref_doc_id or "None",
        "ref_doc_id": node.ref_doc_id or "None",
    }


def test_vector_memory_get_ignores_document_nodes_without_sub_dicts() -> None:
    chroma_client = chromadb.EphemeralClient()
    collection_name = f"vector-memory-{uuid.uuid4()}"
    collection = chroma_client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = MockEmbedding(embed_dim=3)

    doc_node = TextNode(
        id_="doc-node",
        text="normal document chunk",
        metadata={"source": "doc.txt"},
        embedding=[0.0, 0.0, 1.0],
    )

    chat_msg = ChatMessage.from_str("hello from chat", "user")
    chat_node = TextNode(
        id_="chat-node",
        text="hello from chat",
        metadata={"sub_dicts": [chat_msg.model_dump()]},
        embedding=[0.0, 1.0, 0.0],
    )

    collection.add(
        ids=[doc_node.node_id, chat_node.node_id],
        embeddings=[doc_node.get_embedding(), chat_node.get_embedding()],
        documents=[doc_node.text, chat_node.text],
        metadatas=[
            _node_to_chroma_metadata(doc_node),
            _node_to_chroma_metadata(chat_node),
        ],
    )

    mem = VectorMemory.from_defaults(
        vector_store=vector_store,
        embed_model=embed_model,
        retriever_kwargs={"similarity_top_k": 2},
    )

    # Should not raise KeyError even if a retrieved node lacks `sub_dicts`.
    result = mem.get("any query")

    assert len(result) == 1
    assert result[0].content == "hello from chat"
