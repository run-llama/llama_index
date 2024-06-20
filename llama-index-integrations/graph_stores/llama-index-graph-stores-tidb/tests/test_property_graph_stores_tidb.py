import os
from unittest import TestCase, SkipTest

from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
)

from llama_index.graph_stores.tidb import TiDBPropertyGraphStore


def get_store():
    return TiDBPropertyGraphStore(
        db_connection_string=os.environ.get("TIDB_TEST_CONNECTION_STRING"),
        drop_existing_table=True,
        relation_table_name="test_relations",
        node_table_name="test_nodes",
    )


class TestTiDBPropertyGraphStore(TestCase):
    @classmethod
    def setUp(self) -> None:
        try:
            get_store()
        except Exception:
            raise SkipTest("TiDB cluster is not available")

        self.e1 = EntityNode(name="e1", properties={"p1": "v1"})
        self.e2 = EntityNode(name="e2")
        self.r = Relation(label="r", source_id=self.e1.id, target_id=self.e2.id)

    def test_add(self):
        g = get_store()

        g.upsert_nodes([self.e1, self.e2])
        g.upsert_relations([self.r])
        assert len(g.get_triplets(entity_names=["e1"])) == 1
        assert len(g.get_triplets(entity_names=["e3"])) == 0
        assert len(g.get_triplets(properties={"p1": "v1"})) == 1
        assert len(g.get_triplets(properties={"p1": "v2"})) == 0

    def test_delete(self):
        g = get_store()

        g.upsert_nodes([self.e1, self.e2])
        g.upsert_relations([self.r])
        assert len(g.get_triplets(entity_names=["e1"])) == 1
        g.delete(entity_names=["e1"])
        assert len(g.get_triplets(entity_names=["e1"])) == 0

    def test_get(self):
        g = get_store()

        g.upsert_nodes([self.e1, self.e2])
        assert len(g.get(ids=[self.e1.id])) == 1
        assert len(g.get(ids=[self.e1.id, self.e2.id])) == 2
        assert len(g.get(properties={"p1": "v1"})) == 1
