import os
import pickle
import types

import pytest

from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.txtai import TxtaiVectorStore


def test_class():
    names_of_base_classes = [b.__name__ for b in TxtaiVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def _install_fake_txtai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a minimal `txtai` module so tests don't need the real dependency."""

    txtai = types.ModuleType("txtai")
    ann = types.ModuleType("txtai.ann")

    class ANN:  # noqa: D401 - simple dummy protocol for typing/cast
        pass

    class _FakeIndex:
        def __init__(self, config: object) -> None:
            self.config = config
            self.loaded_path: str | None = None

        def load(self, path: str) -> None:
            self.loaded_path = path

    class ANNFactory:
        @staticmethod
        def create(config: object) -> _FakeIndex:
            return _FakeIndex(config)

    ann.ANN = ANN
    ann.ANNFactory = ANNFactory
    txtai.ann = ann

    monkeypatch.setitem(__import__("sys").modules, "txtai", txtai)
    monkeypatch.setitem(__import__("sys").modules, "txtai.ann", ann)


def test_from_persist_path_allows_safe_pickle_config(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    _install_fake_txtai(monkeypatch)

    persist_path = tmp_path / "vector_store.json"
    persist_path.write_text("{}", encoding="utf-8")

    config_path = tmp_path / "config"
    with open(config_path, "wb") as f:
        pickle.dump({"backend": "numpy"}, f, protocol=pickle.HIGHEST_PROTOCOL)

    store = TxtaiVectorStore.from_persist_path(str(persist_path), fs=None)
    assert isinstance(store, TxtaiVectorStore)


def test_from_persist_path_rejects_malicious_pickle_config(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    _install_fake_txtai(monkeypatch)

    persist_path = tmp_path / "vector_store.json"
    persist_path.write_text("{}", encoding="utf-8")

    # If this payload were unpickled with pickle.load(), it would execute
    # os.system(). Our loader should reject it before execution.
    class _Exploit:
        def __reduce__(self):
            return (os.system, ("echo exploited",))

    config_path = tmp_path / "config"
    with open(config_path, "wb") as f:
        pickle.dump(_Exploit(), f, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(pickle.UnpicklingError):
        TxtaiVectorStore.from_persist_path(str(persist_path), fs=None)
