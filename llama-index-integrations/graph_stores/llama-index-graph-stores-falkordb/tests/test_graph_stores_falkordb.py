import os
import time
import docker
import unittest
from llama_index.graph_stores.falkordb.base import FalkorDBGraphStore

# Set up FalkorDB URL
falkordb_url = os.getenv("FALKORDB_TEST_URL", "redis://localhost:6379")

# Set up Docker client and container
docker_client = docker.from_env()


@unittest.skipIf(falkordb_url is None, "No FalkorDB URL provided")
class TestFalkorDBGraphStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start FalkorDB container
        cls.container = docker_client.containers.run(
            "falkordb/falkordb:latest",
            detach=True,
            name="falkordb_test_instance",
            ports={"6379/tcp": 6379},
        )
        time.sleep(2)  # Allow time for the container to initialize
        cls.graph_store = FalkorDBGraphStore(url=falkordb_url)

    @classmethod
    def tearDownClass(cls):
        cls.container.stop()
        cls.container.remove()

    def test_upsert_triplet(self):
        # Insert data and verify
        self.graph_store.upsert_triplet("node1", "related_to", "node2")
        result = self.graph_store.get("node1")
        expected_result = ["RELATED_TO", "node2"]
        self.assertIn(expected_result, result)

        # Retrieve relationship map and verify
        result = self.graph_store.get_rel_map(["node1"], 1)
        self.assertIn(expected_result, result["node1"])

        # Delete data and verify deletion
        self.graph_store.delete("node1", "related_to", "node2")
        result = self.graph_store.get("node1")
        self.assertEqual([], result)

        # Test graph switching and schema refresh
        self.graph_store.switch_graph("new_graph")
        self.graph_store.refresh_schema()


if __name__ == "__main__":
    unittest.main()
