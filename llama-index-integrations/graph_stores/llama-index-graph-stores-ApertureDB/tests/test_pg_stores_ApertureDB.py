
import aperturedb.CommonLibrary
from llama_index.core.graph_stores.types import Relation, EntityNode

import aperturedb
import json

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
            response = [{
                "FindEntity": {
                    "returned": 1,
                    "status": 0,
                    "entities": [{"id": "James", "name": "James", "label": "PERSON"}]
                }}
            ]
        else:
            response = [{"FindEntity": {"returned": 0, "status": 0}}]

        print("query response:", response)
        return response, blobs

    def last_query_ok(self):
        return True

    def get_last_response_str(self):
        return "response"

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

def test_ApertureDB_pg_store_data_add(monkeypatch) -> None:
    monkeypatch.setattr(
        aperturedb.CommonLibrary,
        "create_connector",
        lambda *args, **kwargs: MockConnector(False),
    )

    entities, relations = synthetic_data()
    from llama_index.graph_stores.ApertureDB import ApertureDBGraphStore
    pg_store = ApertureDBGraphStore()
    pg_store.upsert_nodes(entities)
    pg_store.upsert_relations(relations)

    #20 = 2 (PERSON) + 4 (DISH) + 8 (INGREDIENT) + 6 (relations)
    assert len(pg_store.client.queries) == 20, json.dumps(pg_store.client.queries, indent=2)

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
