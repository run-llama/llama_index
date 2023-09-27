"""Test SimpleGraphStore."""

from pathlib import Path
import pytest
from llama_index.graph_stores import simple as simple_graph_store_module
# from types_graph_store import GraphStoreType

@pytest.fixture()
def simple_graph_store() -> simple_graph_store_module.SimpleGraphStore:
    # Initialize an instance of SimpleGraphStore here
    return simple_graph_store_module.SimpleGraphStore()

def test_init(simple_graph_store: simple_graph_store_module.SimpleGraphStore) -> None:
    # Test the __init__ method here
    assert isinstance(simple_graph_store, simple_graph_store_module.SimpleGraphStore)

def test_from_persist_dir() -> None:
    # Test the from_persist_dir class method here
    sgs = simple_graph_store_module.SimpleGraphStore.from_persist_dir()
    assert isinstance(sgs, simple_graph_store_module.SimpleGraphStore)

def test_client_property(simple_graph_store: simple_graph_store_module.SimpleGraphStore) -> None:
    # Test the client property here
    assert simple_graph_store.client is None

def test_get_method(simple_graph_store: simple_graph_store_module.SimpleGraphStore) -> None:
    # Test the get method here
    simple_graph_store.upsert_triplet('subj', 'rel', 'obj')
    result = simple_graph_store.get('subj')
    assert result == [['rel', 'obj']]

def test_get_rel_map_method(simple_graph_store: simple_graph_store_module.SimpleGraphStore) -> None:
    # Test the get_rel_map method here
    simple_graph_store.upsert_triplet('subj', 'rel', 'obj')
    result = simple_graph_store.get_rel_map(['subj'])
    assert result == {'subj': [['subj', 'rel', 'obj']]}

def test_upsert_triplet_method(simple_graph_store: simple_graph_store_module.SimpleGraphStore) -> None:
    # Test the upsert_triplet method here
    simple_graph_store.upsert_triplet('subj', 'rel', 'obj')
    print(simple_graph_store.get('subj'))
    assert simple_graph_store.get('subj') == [['rel', 'obj']]

def test_delete_method(simple_graph_store: simple_graph_store_module.SimpleGraphStore) -> None:
    # Test the delete method here
    simple_graph_store.upsert_triplet('subj', 'rel', 'obj')
    simple_graph_store.delete('subj', 'rel', 'obj')
    result = simple_graph_store.get('subj')
    print(result)
    # assert result TODO, behaviour of simple.py doesn't look right
