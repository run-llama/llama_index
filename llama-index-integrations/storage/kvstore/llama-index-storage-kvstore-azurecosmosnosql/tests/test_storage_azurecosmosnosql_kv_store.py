from unittest.mock import MagicMock, create_autospec

import pytest
from azure.cosmos import ContainerProxy
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from llama_index.core.storage.kvstore.types import BaseKVStore

from llama_index.storage.kvstore.azurecosmosnosql import AzureCosmosNoSqlKVStore


def test_class():
    names_of_base_classes = [b.__name__ for b in AzureCosmosNoSqlKVStore.__mro__]
    assert BaseKVStore.__name__ in names_of_base_classes


def _store(container):
    store = AzureCosmosNoSqlKVStore.__new__(AzureCosmosNoSqlKVStore)
    object.__setattr__(store, "_container", container)
    return store


def _not_found():
    return CosmosResourceNotFoundError(
        message="Entity with the specified id does not exist."
    )


class TestDelete:
    def test_delete_passes_the_partition_key(self):
        """delete_item requires partition_key, so omitting it raises TypeError on every call."""
        container = MagicMock()
        store = _store(container)

        assert store.delete("k") is True
        container.delete_item.assert_called_once_with("k", partition_key="k")

    def test_missing_key_is_false(self):
        container = MagicMock()
        container.delete_item.side_effect = _not_found()

        assert _store(container).delete("k") is False

    def test_a_backend_failure_is_not_a_missing_key(self):
        """False means the key was absent. Every sibling KV store propagates anything else."""
        container = MagicMock()
        container.delete_item.side_effect = RuntimeError("429 request rate is large")

        with pytest.raises(RuntimeError, match="429"):
            _store(container).delete("k")


class TestGet:
    def test_get_passes_the_partition_key(self):
        container = MagicMock()
        container.read_item.return_value = {"id": "k", "messages": {"a": 1}}
        store = _store(container)

        assert store.get("k") == {"a": 1}
        container.read_item.assert_called_once_with("k", partition_key="k")

    def test_missing_key_is_none(self):
        container = MagicMock()
        container.read_item.side_effect = _not_found()

        assert _store(container).get("k") is None

    def test_a_backend_failure_propagates(self):
        container = MagicMock()
        container.read_item.side_effect = RuntimeError("401 unauthorized")

        with pytest.raises(RuntimeError, match="401"):
            _store(container).get("k")


class TestAgainstTheRealSignature:
    """
    A bare MagicMock accepts any arguments, so it cannot show why this mattered.
    Autospec enforces azure-cosmos' real signature, where partition_key is a required
    positional on both read_item and delete_item across the supported range
    (azure-cosmos>=4.7.0,<5).
    """

    def test_delete_against_the_real_signature(self):
        container = create_autospec(ContainerProxy, instance=True)
        assert _store(container).delete("k") is True

    def test_get_against_the_real_signature(self):
        container = create_autospec(ContainerProxy, instance=True)
        container.read_item.return_value = {"id": "k", "messages": {"a": 1}}
        assert _store(container).get("k") == {"a": 1}
