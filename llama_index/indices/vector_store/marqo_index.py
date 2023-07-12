from typing import Any, Dict, List, Optional, Sequence, Tuple

from llama_index.async_utils import run_async_tasks
from llama_index.data_structs.data_structs import IndexDict
from llama_index.indices.base import BaseIndex
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.service_context import ServiceContext
from llama_index.schema import BaseNode, ImageNode, IndexNode, MetadataMode
from llama_index.storage.docstore.types import RefDocInfo
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.types import NodeWithEmbedding, VectorStore
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.marqo import MarqoVectorStore 

class MarqoVectorStoreIndex(VectorStoreIndex):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self._vector_store, MarqoVectorStore):
            raise ValueError("Vector store must be an instance of MarqoVectorStore.")

    def _add_nodes_to_index(self, index_struct: IndexDict, nodes: Sequence[BaseNode], show_progress: bool = False) -> None:
        if not nodes:
            return

        # Get the id and content for each node
        documents = [(n.node_id, n.get_content(metadata_mode=MetadataMode.EMBED)) for n in nodes]

        # Call the vector store's add method with the documents
        new_ids = self._vector_store.add(documents)

        # If the vector store doesn't store text, we need to add the nodes to the
        # index struct and document store
        if not self._vector_store.stores_text or self._store_nodes_override:
            for node, new_id in zip(nodes, new_ids):
                index_struct.add_node(node, text_id=new_id)
                self._docstore.add_documents([node], allow_update=True)
        else:
            # If the vector store keeps text, we only need to add image and index nodes
            for node, new_id in zip(nodes, new_ids):
                if isinstance(node, (ImageNode, IndexNode)):
                    index_struct.add_node(node, text_id=new_id)
                    self._docstore.add_documents([node], allow_update=True)
