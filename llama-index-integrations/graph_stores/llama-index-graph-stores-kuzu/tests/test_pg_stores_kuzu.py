import shutil
import pytest

from llama_index.graph_stores.kuzu import KuzuPropertyGraphStore
from llama_index.core.graph_stores.types import Relation, EntityNode
from llama_index.core.schema import TextNode


@pytest.fixture()
def pg_store() -> KuzuPropertyGraphStore:
    import kuzu

    shutil.rmtree("llama_test_db", ignore_errors=True)
    db = kuzu.Database("llama_test_db")
    pg_store = KuzuPropertyGraphStore(db)
    pg_store.structured_query("MATCH (n) DETACH DELETE n")
    return pg_store


def test_kuzudb_pg_store(pg_store: KuzuPropertyGraphStore) -> None:
    # Create a two entity nodes
    entity1 = EntityNode(label="PERSON", name="Logan")
    entity2 = EntityNode(label="ORGANIZATION", name="LlamaIndex")

    # Create a relation
    relation = Relation(
        label="WORKS_FOR",
        source_id=entity1.id,
        target_id=entity2.id,
    )

    pg_store.upsert_nodes([entity1, entity2])
    pg_store.upsert_relations([relation])

    source_node = TextNode(text="Logan (age 28), works for LlamaIndex since 2023.")
    relations = [
        Relation(
            label="MENTIONS",
            target_id=entity1.id,
            source_id=source_node.node_id,
        ),
        Relation(
            label="MENTIONS",
            target_id=entity2.id,
            source_id=source_node.node_id,
        ),
    ]

    pg_store.upsert_llama_nodes([source_node])
    pg_store.upsert_relations(relations)

    print(pg_store.get())

    kg_nodes = pg_store.get(ids=[entity1.id])
    assert len(kg_nodes) == 1
    assert kg_nodes[0].label == "PERSON"
    assert kg_nodes[0].name == "Logan"

    # get paths from a node
    paths = pg_store.get_rel_map(kg_nodes, depth=1)
    for path in paths:
        assert path[0].id == entity1.id
        assert path[2].id == entity2.id
        assert path[1].id == relation.id

    query = "match (n:Entity) return n"
    result = pg_store.structured_query(query)
    assert len(result) == 2

    # deleting
    # delete our entities
    pg_store.delete(ids=[entity1.id, entity2.id])

    # delete our text nodes
    pg_store.delete(ids=[source_node.node_id])

    nodes = pg_store.get(ids=[entity1.id, entity2.id])
    assert len(nodes) == 0

    text_nodes = pg_store.get_llama_nodes([source_node.node_id])
    assert len(text_nodes) == 0

    # Delete the database
    shutil.rmtree("llama_test_db", ignore_errors=True)
