import builtins
import sys
import types
from unittest.mock import MagicMock

from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.storage.storage_context import StorageContext


def test_load_from_disk_closes_multi_embed_store_file_handle(tmp_path, monkeypatch):
    fake_flagembedding = types.ModuleType("FlagEmbedding")
    fake_flagembedding.BGEM3FlagModel = MagicMock()
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_flagembedding)

    from llama_index.indices.managed.bge_m3 import BGEM3Index

    persist_dir = tmp_path / "storage"
    sc = StorageContext.from_defaults()
    sc.index_store.add_index_struct(IndexDict())
    sc.persist(persist_dir=str(persist_dir))

    import pickle

    with open(persist_dir / "multi_embed_store.pkl", "wb") as f:
        pickle.dump({"dense": []}, f)

    opened_files = []
    real_open = builtins.open

    def tracking_open(*args, **kwargs):
        f = real_open(*args, **kwargs)
        opened_files.append(f)
        return f

    monkeypatch.setattr(builtins, "open", tracking_open)
    index = BGEM3Index.load_from_disk(persist_dir=str(persist_dir))

    assert index._multi_embed_store == {"dense": []}
    read_handles = [f for f in opened_files if "multi_embed_store.pkl" in f.name]
    assert read_handles
    assert all(f.closed for f in read_handles)
