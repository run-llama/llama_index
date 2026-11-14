"""A failed local snapshot write must not destroy the previous snapshot."""

from contextlib import contextmanager

import fsspec
import pytest

from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode
from llama_index.core.schema import TextNode
from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore
from llama_index.core.vector_stores.simple import SimpleVectorStore


@pytest.fixture(params=["kv", "vector", "graph", "property_graph"])
def snapshots(request):
    """Two distinct, valid snapshots through the public store APIs."""
    if request.param == "kv":
        old, new = SimpleKVStore(), SimpleKVStore()
        old.put("old", {"value": "previous"})
        new.put("new", {"value": "replacement"})
    elif request.param == "vector":
        old, new = SimpleVectorStore(), SimpleVectorStore()
        old.add([TextNode(id_="old", text="previous", embedding=[1.0, 0.0])])
        new.add([TextNode(id_="new", text="replacement", embedding=[0.0, 1.0])])
    elif request.param == "graph":
        old, new = SimpleGraphStore(), SimpleGraphStore()
        old.upsert_triplet("old", "is", "previous")
        new.upsert_triplet("new", "is", "replacement")
    else:
        old, new = SimplePropertyGraphStore(), SimplePropertyGraphStore()
        old.upsert_nodes([EntityNode(name="old")])
        new.upsert_nodes([EntityNode(name="new")])
    return old, new


class PartialWriter:
    def __init__(self, stream, during_write):
        self.stream = stream
        self.during_write = during_write

    def write(self, text):
        self.stream.write(text[:1])
        self.stream.flush()
        self.during_write()
        return self.stream.write(text[1:]) + min(1, len(text))


@contextmanager
def intercept_writes(fs, monkeypatch, during_write):
    """Inject at the actual filesystem write, keeping files and readers real."""
    real_open = fs.open

    @contextmanager
    def wrap(stream):
        with stream:
            yield PartialWriter(stream, during_write)

    def open_file(path, mode="rb", **kwargs):
        stream = real_open(path, mode, **kwargs)
        return wrap(stream) if mode in ("w", "x") else stream

    with monkeypatch.context() as patch:
        patch.setattr(fs, "open", open_file)
        yield


def test_failed_write_keeps_previous_snapshot(tmp_path, snapshots, monkeypatch):
    old, new = snapshots
    path = tmp_path / "snapshot.json"
    fs = fsspec.filesystem("file", skip_instance_cache=True)
    old.persist(str(path), fs=fs)
    previous = path.read_bytes()

    def fail():
        raise OSError("injected disk write failure")

    with intercept_writes(fs, monkeypatch, fail):
        with pytest.raises(OSError, match="injected disk write failure"):
            new.persist(str(path), fs=fs)

    assert path.read_bytes() == previous
    assert type(old).from_persist_path(str(path)).to_dict() == old.to_dict()
    assert list(tmp_path.iterdir()) == [path]


def test_reader_sees_previous_snapshot_until_write_finishes(
    tmp_path, snapshots, monkeypatch
):
    old, new = snapshots
    path = tmp_path / "snapshot.json"
    fs = fsspec.filesystem("file", skip_instance_cache=True)
    old.persist(str(path), fs=fs)
    previous = path.read_bytes()

    def read_during_write():
        assert path.read_bytes() == previous
        assert type(old).from_persist_path(str(path)).to_dict() == old.to_dict()

    with intercept_writes(fs, monkeypatch, read_during_write):
        new.persist(str(path), fs=fs)

    assert type(new).from_persist_path(str(path)).to_dict() == new.to_dict()
    assert list(tmp_path.iterdir()) == [path]


def test_serialization_failure_keeps_previous_kv_snapshot(tmp_path):
    path = tmp_path / "snapshot.json"
    store = SimpleKVStore()
    store.put("old", {"value": "previous"})
    store.persist(str(path))
    previous = path.read_bytes()
    store.put("unsupported", {"value": object()})

    with pytest.raises(TypeError):
        store.persist(str(path))

    assert path.read_bytes() == previous
    assert SimpleKVStore.from_persist_path(str(path)).get("old") == {
        "value": "previous"
    }
    assert list(tmp_path.iterdir()) == [path]


def test_failed_first_write_does_not_publish_partial_file(
    tmp_path, snapshots, monkeypatch
):
    _, new = snapshots
    path = tmp_path / "snapshot.json"
    fs = fsspec.filesystem("file", skip_instance_cache=True)

    def fail():
        raise OSError("injected disk write failure")

    with intercept_writes(fs, monkeypatch, fail):
        with pytest.raises(OSError, match="injected disk write failure"):
            new.persist(str(path), fs=fs)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["close", "replace"])
def test_failed_publication_keeps_previous_snapshot(
    tmp_path, snapshots, monkeypatch, failure
):
    import os

    old, new = snapshots
    path = tmp_path / "snapshot.json"
    fs = fsspec.filesystem("file", skip_instance_cache=True)
    old.persist(str(path), fs=fs)
    previous = path.read_bytes()

    if failure == "replace":

        def fail_replace(source, destination):
            raise OSError("injected publication failure")

        monkeypatch.setattr(os, "replace", fail_replace)
    else:
        real_open = fs.open

        @contextmanager
        def fail_close(stream):
            with stream:
                yield stream
            raise OSError("injected publication failure")

        def open_file(path, mode="rb", **kwargs):
            stream = real_open(path, mode, **kwargs)
            return fail_close(stream) if mode in ("w", "x") else stream

        monkeypatch.setattr(fs, "open", open_file)

    with pytest.raises(OSError, match="injected publication failure"):
        new.persist(str(path), fs=fs)

    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_remote_filesystem_round_trip(snapshots):
    from uuid import uuid4

    _, new = snapshots
    fs = fsspec.filesystem("memory")
    path = f"/{uuid4().hex}/snapshot.json"
    try:
        new.persist(path, fs=fs)
        assert type(new).from_persist_path(path, fs=fs).to_dict() == new.to_dict()
    finally:
        fs.rm(path.rsplit("/", 1)[0], recursive=True)


@pytest.mark.parametrize("commit", [True, False])
def test_caller_transaction_controls_publication(tmp_path, snapshots, commit):
    old, new = snapshots
    path = tmp_path / "snapshot.json"
    fs = fsspec.filesystem("file", skip_instance_cache=True)
    old.persist(str(path), fs=fs)
    previous = path.read_bytes()

    class Abort(Exception):
        pass

    try:
        with fs.transaction:
            new.persist(str(path), fs=fs)
            assert path.read_bytes() == previous
            if not commit:
                raise Abort
    except Abort:
        pass

    expected = new if commit else old
    assert type(expected).from_persist_path(str(path)).to_dict() == expected.to_dict()
    assert list(tmp_path.iterdir()) == [path]


def test_local_file_url_and_automatic_directories(tmp_path, snapshots):
    _, new = snapshots
    path = tmp_path / "nested" / "snapshot.json"
    fs = fsspec.filesystem("file", auto_mkdir=True, skip_instance_cache=True)
    new.persist(path.as_uri(), fs=fs)
    assert type(new).from_persist_path(path.as_uri(), fs=fs).to_dict() == new.to_dict()
    assert list(path.parent.iterdir()) == [path]


def test_symlink_still_updates_target(tmp_path, snapshots):
    old, new = snapshots
    target = tmp_path / "target.json"
    link = tmp_path / "link.json"
    old.persist(str(target))
    try:
        link.symlink_to(target.name)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    new.persist(str(link))
    assert link.is_symlink()
    assert type(new).from_persist_path(str(target)).to_dict() == new.to_dict()


def test_existing_file_permissions_are_preserved(tmp_path, snapshots):
    import os
    import stat

    if os.name != "posix":
        pytest.skip("POSIX permission bits")
    old, new = snapshots
    path = tmp_path / "snapshot.json"
    old.persist(str(path))
    path.chmod(0o640)
    new.persist(str(path))
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_read_only_file_is_not_replaced(tmp_path, snapshots):
    import os

    if os.name != "posix" or os.geteuid() == 0:
        pytest.skip("requires an unprivileged POSIX user")
    old, new = snapshots
    path = tmp_path / "snapshot.json"
    old.persist(str(path))
    previous = path.read_bytes()
    path.chmod(0o444)
    try:
        with pytest.raises(PermissionError):
            new.persist(str(path))
        assert path.read_bytes() == previous
        assert list(tmp_path.iterdir()) == [path]
    finally:
        path.chmod(0o644)


def test_private_snapshot_remains_private_during_write(tmp_path, monkeypatch):
    import os
    import stat

    if os.name != "posix":
        pytest.skip("POSIX permission bits")
    path = tmp_path / "snapshot.json"
    store = SimpleKVStore()
    store.put("secret", {"value": "old secret"})
    store.persist(str(path))
    path.chmod(0o600)
    store.put("secret", {"value": "new secret"})
    fs = fsspec.filesystem("file", skip_instance_cache=True)

    def inspect_staged_file():
        staged = list(tmp_path.glob(".llama-*.tmp"))
        assert len(staged) == 1
        assert stat.S_IMODE(staged[0].stat().st_mode) == 0o600

    previous_umask = os.umask(0o022)
    try:
        with intercept_writes(fs, monkeypatch, inspect_staged_file):
            store.persist(str(path), fs=fs)
    finally:
        os.umask(previous_umask)


def test_custom_local_filesystem_keeps_its_write_semantics(tmp_path, snapshots):
    from fsspec.implementations.local import LocalFileSystem

    class CustomLocalFileSystem(LocalFileSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.written_paths = []

        def _open(self, path, mode="rb", **kwargs):
            if "w" in mode:
                self.written_paths.append(path)
            return super()._open(path, mode=mode, **kwargs)

    _, new = snapshots
    path = tmp_path / "snapshot.json"
    fs = CustomLocalFileSystem(skip_instance_cache=True)
    new.persist(str(path), fs=fs)
    assert fs.written_paths == [str(path)]
    assert type(new).from_persist_path(str(path), fs=fs).to_dict() == new.to_dict()


def test_cleanup_failure_does_not_mask_write_failure(tmp_path, monkeypatch, caplog):
    import os

    path = tmp_path / "snapshot.json"
    store = SimpleKVStore()
    store.put("old", {"value": "previous"})
    store.persist(str(path))
    previous = path.read_bytes()
    store.put("new", {"value": "replacement"})
    fs = fsspec.filesystem("file", skip_instance_cache=True)

    def fail_write():
        raise OSError("original disk write failure")

    def fail_cleanup(path):
        raise PermissionError("injected cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "unlink", fail_cleanup)
        with intercept_writes(fs, monkeypatch, fail_write):
            with pytest.raises(OSError, match="original disk write failure"):
                store.persist(str(path), fs=fs)

    assert path.read_bytes() == previous
    assert "Could not remove staged snapshot" in caplog.text
    staged = list(tmp_path.glob(".llama-*.tmp"))
    assert len(staged) == 1
    staged[0].unlink()
