import inspect

from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.core.graph_stores.types import GraphStore, PropertyGraphStore


def test_simple_graph_store_query_param_map_default_is_none() -> None:
    """The mutable-default bug: param_map must default to None, not {}."""
    sig = inspect.signature(SimpleGraphStore.query)
    assert sig.parameters["param_map"].default is None


def test_graph_store_query_param_map_default_is_none() -> None:
    """The abstract GraphStore.query must default param_map to None."""
    sig = inspect.signature(GraphStore.query)
    assert sig.parameters["param_map"].default is None


def test_property_graph_store_astructured_query_param_map_default_is_none() -> None:
    """PropertyGraphStore.astructured_query must default param_map to None."""
    sig = inspect.signature(PropertyGraphStore.astructured_query)
    assert sig.parameters["param_map"].default is None


class _QueryStoreWithMutableParamMap(SimpleGraphStore):
    """Concrete store whose query() mutates param_map, to detect shared state."""

    def query(self, query: str, param_map: dict | None = None) -> dict:
        param_map = param_map or {}
        param_map["called"] = True
        return param_map


def test_query_twice_without_param_map_does_not_share_state() -> None:
    """Each call that omits param_map gets a fresh dict (no leaked state)."""
    store = _QueryStoreWithMutableParamMap()
    result1 = store.query("SELECT 1")
    result2 = store.query("SELECT 2")

    assert result1 is not result2
    assert result1 == {"called": True}
    assert result2 == {"called": True}


def test_query_with_explicit_param_map_unchanged() -> None:
    """An explicitly passed param_map is used as-is (behavior unchanged)."""
    store = _QueryStoreWithMutableParamMap()
    explicit = {"foo": "bar"}
    result = store.query("SELECT 1", param_map=explicit)

    assert result is explicit
    assert result == {"foo": "bar", "called": True}
