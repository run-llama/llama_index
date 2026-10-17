"""
Regression tests for B006 (mutable default argument) in the graph store
`query`/`astructured_query` stubs. `param_map: Optional[Dict] = {}` shares
one dict object across every call that relies on the default - harmless
today since these particular stubs don't mutate it, but the anti-pattern
is a real trap for the first subclass override that does.
"""

import inspect

from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.core.graph_stores.types import GraphStore, PropertyGraphStore


def test_simple_graph_store_query_uses_none_default() -> None:
    sig = inspect.signature(SimpleGraphStore.query)
    assert sig.parameters["param_map"].default is None


def test_graph_store_query_uses_none_default() -> None:
    sig = inspect.signature(GraphStore.query)
    assert sig.parameters["param_map"].default is None


def test_property_graph_store_astructured_query_uses_none_default() -> None:
    sig = inspect.signature(PropertyGraphStore.astructured_query)
    assert sig.parameters["param_map"].default is None
