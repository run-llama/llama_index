from llama_index.data_structs.data_structs import IndexGraph
from llama_index.storage.index_store.simple_index_store import SimpleIndexStore


def test_simple_index_store_dict() -> None:
    index_struct = IndexGraph()
    index_store = SimpleIndexStore()
    index_store.add_index_struct(index_struct)

    # save
    save_dict = index_store.to_dict()

    # load
    loaded_index_store = SimpleIndexStore.from_dict(save_dict)

    # test
    assert loaded_index_store.get_index_struct(index_struct.index_id) == index_struct
