import aperturedb.CommonLibrary
from llama_index.core.graph_stores.types import Relation, EntityNode
from llama_index.graph_stores.ApertureDB import ApertureDBGraphStore


import aperturedb
import json

import pytest


@pytest.fixture
def create_store(monkeypatch):
    class MockConnector:
        def __init__(self, exists) -> None:
            self.exists = exists
            self.queries = []
            print(f"self.exists: {self.exists}")

        def clone(self):
            return self

        def query(self, *args, **kwargs):
            print("query called with args:", args, "and kwargs:", kwargs)
            self.queries.append(args[0])

            response, blobs = [], []
            if self.exists:
                response = [
                    {
                        "FindEntity": {
                            "returned": 1,
                            "status": 0,
                            "entities": [
                                {"id": "James", "name": "James", "label": "PERSON"}
                            ],
                        }
                    }
                ]
            else:
                response = [{"FindEntity": {"returned": 0, "status": 0}}]

            print("query response:", response)
            return response, blobs

        def last_query_ok(self):
            return True

        def get_last_response_str(self):
            return "response"

    def store_creator(data_exists):
        monkeypatch.setattr(
            aperturedb.CommonLibrary,
            "create_connector",
            lambda *args, **kwargs: MockConnector(data_exists),
        )

        return ApertureDBGraphStore()

    return store_creator


@pytest.fixture
def synthetic_data():
    entities = [
        EntityNode(label="PERSON", name="James"),
        EntityNode(label="DISH", name="Butter Chicken"),
        EntityNode(label="DISH", name="Scrambled Eggs"),
        EntityNode(label="INGREDIENT", name="Butter"),
        EntityNode(label="INGREDIENT", name="Chicken"),
        EntityNode(label="INGREDIENT", name="Eggs"),
        EntityNode(label="INGREDIENT", name="Salt"),
    ]

    relations = [
        Relation(
            label="EATS",
            source_id=entities[0].id,
            target_id=entities[1].id,
        ),
        Relation(
            label="EATS",
            source_id=entities[0].id,
            target_id=entities[2].id,
        ),
        Relation(
            label="CONTAINS",
            source_id=entities[1].id,
            target_id=entities[3].id,
        ),
        Relation(
            label="HAS",
            source_id=entities[1].id,
            target_id=entities[4].id,
        ),
        Relation(
            label="COMPRISED_OF",
            source_id=entities[2].id,
            target_id=entities[5].id,
        ),
        Relation(
            label="GOT",
            source_id=entities[2].id,
            target_id=entities[6].id,
        ),
    ]

    return entities, relations


def test_ApertureDB_pg_store_data_update(create_store, synthetic_data) -> None:
    pg_store = create_store(True)
    entities, relations = synthetic_data
    pg_store.upsert_nodes(entities)
    pg_store.upsert_relations(relations)

    # 20 = 2 (PERSON) + 4 (DISH) + 8 (INGREDIENT) + 6 (relations)
    assert len(pg_store.client.queries) == 20, json.dumps(
        pg_store.client.queries, indent=2
    )

    # Check if the queries are correct, FindEntity followed by AddEntity.
    for i in range(14):
        q = pg_store.client.queries[i]
        assert len(q) == 1, json.dumps(q, indent=2)
        if i % 2 == 0:
            assert "FindEntity" in q[0]
        else:
            assert "UpdateEntity" in q[0]
    # Check if the queries are correct, FindEntity x 2 followed by AddConnection.
    for i in range(14, 20):
        q = pg_store.client.queries[i]
        assert len(q) == 3, json.dumps(q, indent=2)
        assert "FindEntity" in q[0]
        assert "FindEntity" in q[1]
        assert "AddConnection" in q[2]


def test_ApertureDB_pg_store_data_add(create_store, synthetic_data) -> None:
    pg_store = create_store(False)
    entities, relations = synthetic_data
    pg_store.upsert_nodes(entities)
    pg_store.upsert_relations(relations)

    # 20 = 2 (PERSON) + 4 (DISH) + 8 (INGREDIENT) + 6 (relations)
    assert len(pg_store.client.queries) == 20, json.dumps(
        pg_store.client.queries, indent=2
    )

    # Check if the queries are correct, FindEntity followed by AddEntity.
    for i in range(14):
        q = pg_store.client.queries[i]
        assert len(q) == 1, json.dumps(q, indent=2)
        if i % 2 == 0:
            assert "FindEntity" in q[0]
        else:
            assert "AddEntity" in q[0]
    # Check if the queries are correct, FindEntity x 2 followed by AddConnection.
    for i in range(14, 20):
        q = pg_store.client.queries[i]
        assert len(q) == 3, json.dumps(q, indent=2)
        assert "FindEntity" in q[0]
        assert "FindEntity" in q[1]
        assert "AddConnection" in q[2]


def test_ApertureDB_pg_store_delete(create_store, synthetic_data) -> None:
    entities, relations = synthetic_data
    pg_store = create_store(True)
    pg_store.upsert_nodes(entities)
    pg_store.upsert_relations(relations)

    # 20 = 2 (PERSON) + 4 (DISH) + 8 (INGREDIENT) + 6 (relations)
    assert len(pg_store.client.queries) == 20, json.dumps(
        pg_store.client.queries, indent=2
    )

    pg_store.client.queries = []
    pg_store.delete(ids=[e.id for e in entities])
    assert len(pg_store.client.queries) == 1, json.dumps(
        pg_store.client.queries, indent=2
    )
    assert "DeleteEntity" in pg_store.client.queries[0][0]
    assert "results" not in pg_store.client.queries[0][0]["DeleteEntity"]
    delete_query_constraints = pg_store.client.queries[0][0]["DeleteEntity"][
        "constraints"
    ]
    assert len(delete_query_constraints) == 1, json.dumps(
        delete_query_constraints, indent=2
    )


def test_ApertureDB_pg_store_structured_query(create_store, synthetic_data) -> None:
    entities, relations = synthetic_data
    pg_store = create_store(True)

    pg_store.structured_query(
        "FindEntity", {"constraints": [{"name": ["==", "James"]}]}
    )
    assert len(pg_store.client.queries) == 1, json.dumps(
        pg_store.client.queries, indent=2
    )
