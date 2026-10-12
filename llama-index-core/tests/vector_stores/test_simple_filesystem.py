import json
from pathlib import Path
from typing import Optional

import fsspec
import pytest
from pytest_mock import MockerFixture

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.simple import SimpleVectorStore


@pytest.mark.parametrize("protocol", [None, "file", "memory"])
def test_load_all_namespaces(tmp_path: Path, protocol: Optional[str]) -> None:
    fs = fsspec.filesystem(protocol) if protocol else None
    persist_dir = str(tmp_path / "parent__with_separator" / "stores")
    expected = {}
    for namespace in ["default", "image"]:
        store = SimpleVectorStore(fs=fs)
        store.add([TextNode(id_=namespace, embedding=[1.0, 2.0])])
        store.persist(f"{persist_dir}/{namespace}__vector_store.json")
        expected[namespace] = store.to_dict()

    restored = SimpleVectorStore.from_namespaced_persist_dir(persist_dir, fs=fs)

    assert set(restored) == set(expected)
    for namespace, store in restored.items():
        assert store.to_dict() == expected[namespace]


@pytest.mark.parametrize("protocol", ["file", "memory"])
def test_loaded_store_retains_filesystem(tmp_path: Path, protocol: str) -> None:
    fs = fsspec.filesystem(protocol)
    original_path = str(tmp_path / "original.json")
    copied_path = str(tmp_path / "copied.json")
    original = SimpleVectorStore(fs=fs)
    original.add([TextNode(id_="first", embedding=[1.0, 2.0])])
    original.persist(original_path)

    restored = SimpleVectorStore.from_persist_path(original_path, fs=fs)
    restored.add([TextNode(id_="second", embedding=[2.0, 1.0])])
    restored.persist(copied_path)

    assert fs.exists(copied_path)
    assert (
        SimpleVectorStore.from_persist_path(copied_path, fs=fs).to_dict()
        == restored.to_dict()
    )
    if protocol == "memory":
        assert not Path(copied_path).exists()


@pytest.mark.parametrize("protocol", [None, "file", "memory"])
def test_namespace_load_errors_are_not_hidden(
    tmp_path: Path, protocol: Optional[str]
) -> None:
    fs = fsspec.filesystem(protocol or "file")
    persist_dir = str(tmp_path)
    SimpleVectorStore(fs=fs).persist(f"{persist_dir}/default__vector_store.json")
    with fs.open(f"{persist_dir}/image__vector_store.json", "w") as stream:
        stream.write("invalid json")

    with pytest.raises(json.JSONDecodeError):
        SimpleVectorStore.from_namespaced_persist_dir(
            persist_dir, fs=fs if protocol else None
        )


def test_listing_failure_preserves_default_fallback(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    fs = fsspec.filesystem("memory")
    store = SimpleVectorStore(fs=fs)
    store.add([TextNode(id_="default", embedding=[1.0, 2.0])])
    store.persist(f"{tmp_path}/default__vector_store.json")
    mocker.patch.object(fs, "listdir", side_effect=PermissionError("listing denied"))

    restored = SimpleVectorStore.from_namespaced_persist_dir(str(tmp_path), fs=fs)

    assert set(restored) == {"default"}
    assert restored["default"].to_dict() == store.to_dict()
