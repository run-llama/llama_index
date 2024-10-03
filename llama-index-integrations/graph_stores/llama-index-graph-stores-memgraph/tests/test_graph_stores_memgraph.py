import unittest
from llama_index.graph_stores.memgraph import MemgraphGraphStore


class TestMemgraphGraphStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = MemgraphGraphStore(
            username="", password="", url="bolt://localhost:7687"
        )

    def test_connection(self):
        """Test if connection to Memgraph is working."""
        try:
            self.store.client.verify_connectivity()
            connected = True
        except Exception as e:
            connected = False
        self.assertTrue(connected, "Could not connect to Memgraph")

    def test_upsert_triplet(self):
        """Test inserting a triplet into Memgraph."""
        self.store.upsert_triplet("Alice", "KNOWS", "Bob")
        triplets = self.store.get("Alice")
        self.assertIn(["KNOWS", "Bob"], triplets)

    def test_delete_triplet(self):
        """Test deleting a triplet from Memgraph."""
        self.store.delete("Alice", "KNOWS", "Bob")
        triplets = self.store.get("Alice")
        self.assertNotIn(["KNOWS", "Bob"], triplets)

    def test_get_rel_map(self):
        """Test retrieving relationships."""
        self.store.upsert_triplet("Alice", "KNOWS", "Bob")
        rel_map = self.store.get_rel_map(["Alice"], depth=2)
        self.assertIn("Alice", rel_map)
        self.assertIn(["KNOWS", "Bob"], rel_map["Alice"])


if __name__ == "__main__":
    unittest.main()
