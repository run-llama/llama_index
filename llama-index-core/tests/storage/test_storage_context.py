import ntpath
import os
from pathlib import Path

import fsspec
import pytest

from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode
from llama_index.core.schema import TextNode
from llama_index.core.storage.storage_context import StorageContext


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


def test_graph_stores_load_from_remote_fs_with_windows_path_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Graph stores load through a remote filesystem when os.path is ntpath.

    With fs given, every store must build its path with concat_dirs: on Windows,
    os.path.join puts a backslash into the remote key and the file is not found.
    """
    fs = fsspec.filesystem("memory")
    persist_dir = f"bucket/{tmp_path.name}"

    property_graph_store = SimplePropertyGraphStore()
    property_graph_store.upsert_nodes([EntityNode(name="alice")])
    storage_context = StorageContext.from_defaults(
        property_graph_store=property_graph_store
    )
    storage_context.graph_store.upsert_triplet("alice", "knows", "bob")
    storage_context.persist(persist_dir=persist_dir, fs=fs)

    with monkeypatch.context() as m:
        # os.path is ntpath on Windows
        m.setattr(os.path, "join", ntpath.join)
        loaded = StorageContext.from_defaults(persist_dir=persist_dir, fs=fs)

    assert loaded.graph_store.get_rel_map(["alice"]) == {
        "alice": [["alice", "knows", "bob"]]
    }
    assert loaded.property_graph_store is not None
    assert len(loaded.property_graph_store.graph.get_all_nodes()) == 1
