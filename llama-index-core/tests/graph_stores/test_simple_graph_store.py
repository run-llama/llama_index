from pathlib import Path

import fsspec
import pytest
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.core.graph_stores.types import DEFAULT_PERSIST_FNAME


@pytest.mark.parametrize("existing_snapshot", [False, True])
@pytest.mark.parametrize("load_from_dir", [False, True])
def test_restored_store_persists_to_original_filesystem(
    tmp_path: Path, existing_snapshot: bool, load_from_dir: bool
) -> None:
    fs = fsspec.filesystem("memory")
    path = str(tmp_path / DEFAULT_PERSIST_FNAME)
    if existing_snapshot:
        initial = SimpleGraphStore(fs=fs)
        initial.upsert_triplet("before", "rel", "first")
        initial.persist(path)

    if load_from_dir:
        store = SimpleGraphStore.from_persist_dir(str(tmp_path), fs=fs)
    else:
        store = SimpleGraphStore.from_persist_path(path, fs=fs)
    store.upsert_triplet("after", "rel", "second")
    store.persist(path)

    restored = SimpleGraphStore.from_persist_path(path, fs=fs)
    assert restored.get("after") == [["rel", "second"]]
    if existing_snapshot:
        assert restored.get("before") == [["rel", "first"]]
    assert not Path(path).exists()


def test_persist_can_override_restored_filesystem(tmp_path: Path) -> None:
    fs = fsspec.filesystem("memory")
    path = str(tmp_path / DEFAULT_PERSIST_FNAME)
    store = SimpleGraphStore.from_persist_path(path, fs=fs)
    store.upsert_triplet("subject", "rel", "object")
    store.persist(path, fs=fsspec.filesystem("file"))

    restored = SimpleGraphStore.from_persist_path(path)
    assert restored.get("subject") == [["rel", "object"]]
    assert not fs.exists(path)
