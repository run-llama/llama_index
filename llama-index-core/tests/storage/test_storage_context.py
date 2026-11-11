from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.schema import TextNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery


def test_from_defaults_preserves_image_vector_store() -> None:
    image_store = SimpleVectorStore()
    image_store.add([TextNode(id_="image", embedding=[1.0, 0.0])])

    storage_context = StorageContext.from_defaults(vector_stores={"image": image_store})

    result = storage_context.vector_stores["image"].query(
        VectorStoreQuery(query_embedding=[1.0, 0.0])
    )
    assert result.ids == ["image"]


def test_from_defaults_explicit_image_store_overrides_namespace() -> None:
    image_store = SimpleVectorStore()
    image_store.add([TextNode(id_="replacement", embedding=[1.0, 0.0])])

    storage_context = StorageContext.from_defaults(
        vector_stores={"image": SimpleVectorStore()}, image_store=image_store
    )

    result = storage_context.vector_stores["image"].query(
        VectorStoreQuery(query_embedding=[1.0, 0.0])
    )
    assert result.ids == ["replacement"]


def test_from_defaults_adds_missing_image_vector_store() -> None:
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(
        vector_stores={"default": vector_store}
    )

    assert storage_context.vector_store is vector_store
    storage_context.vector_stores["image"].add(
        [TextNode(id_="image", embedding=[1.0, 0.0])]
    )
    result = storage_context.vector_stores["image"].query(
        VectorStoreQuery(query_embedding=[1.0, 0.0])
    )
    assert result.ids == ["image"]


def test_storage_context_dict() -> None:
    storage_context = StorageContext.from_defaults()

    # add
    node = TextNode(text="test", embedding=[0.0, 0.0, 0.0])
    index_struct = IndexDict()
    storage_context.vector_store.add([node])
    storage_context.docstore.add_documents([node])
    storage_context.index_store.add_index_struct(index_struct)
    # Refetch the node from the storage context,
    # as its metadata and hash may have changed.
    retrieved_node = storage_context.docstore.get_document(node.node_id)

    # save
    save_dict = storage_context.to_dict()

    # load
    loaded_storage_context = StorageContext.from_dict(save_dict)

    # test
    assert loaded_storage_context.docstore.get_node(node.node_id) == retrieved_node
    assert (
        storage_context.index_store.get_index_struct(index_struct.index_id)
        == index_struct
    )
