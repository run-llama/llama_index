from llama_index.data_structs.data_structs import IndexDict
from llama_index.data_structs.node import Node
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.types import NodeWithEmbedding


def test_storage_context_dict() -> None:
    storage_context = StorageContext.from_defaults()

    # add
    node = Node("test")
    index_struct = IndexDict()
    storage_context.vector_store.add(
        [NodeWithEmbedding(node=node, embedding=[0.0, 0.0, 0.0])]
    )
    storage_context.docstore.add_documents([node])
    storage_context.index_store.add_index_struct(index_struct)

    # save
    save_dict = storage_context.to_dict()

    # load
    loaded_storage_context = StorageContext.from_dict(save_dict)

    # test
    assert loaded_storage_context.docstore.get_node(node.get_doc_id()) == node
    assert (
        storage_context.index_store.get_index_struct(index_struct.index_id)
        == index_struct
    )
