import os
import unittest
from llama_index.graph_stores.falkordb.base import FalkorDBGraphStore

falkordb_url = os.environ.get("FALKORDB_TEST_URL")


@unittest.skipIf(falkordb_url is None, "No FalkorDB URL provided")
class TestFalkorDBGraphStore(unittest.TestCase):
    def setUp(self):
        self.graph_store = FalkorDBGraphStore(url=falkordb_url)

    def test_upsert_triplet(self):
        # Call the method you want to test
        self.graph_store.upsert_triplet("node1", "related_to", "node2")

        # Check if the data has been inserted correctly
        result = self.graph_store.get("node1")  # Adjust the method to retrieve data
        expected_result = [
            "RELATED_TO",
            "node2",
        ]  # Adjust this based on what you expect
        self.assertIn(expected_result, result)

        result = self.graph_store.get_rel_map(["node1"], 1)

        self.assertIn(expected_result, result["node1"])

        self.graph_store.delete("node1", "related_to", "node2")

        result = self.graph_store.get("node1")  # Adjust the method to retrieve data
        expected_result = []  # Adjust this based on what you expect
        self.assertEqual(expected_result, result)

        self.graph_store.switch_graph("new_graph")
        self.graph_store.refresh_schema()


if __name__ == "__main__":
    unittest.main()
