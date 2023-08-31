from llama_index.data_structs.data_structs import IndexDict
from llama_index.schema import TextNode
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.types import NodeWithEmbedding


def test_storage_context_dict() -> None:
    storage_context = StorageContext.from_defaults()

    # add
    node = TextNode(text="test")
    index_struct = IndexDict()
    storage_context.vector_store.add(
        [NodeWithEmbedding(node=node, embedding=[0.0, 0.0, 0.0])]
    )
    storage_context.docstore.add_documents([node])
    storage_context.index_store.add_index_struct(index_struct)
    # Refetch the node from the storage context, as its metadata and hash may have changed.
    node = storage_context.docstore.get_document(node.node_id)

    # save
    save_dict = storage_context.to_dict()

    # load
    loaded_storage_context = StorageContext.from_dict(save_dict)

    # test
    assert loaded_storage_context.docstore.get_node(node.node_id) == node
    assert (
        storage_context.index_store.get_index_struct(index_struct.index_id)
        == index_struct
    )
