import time
import docker
import unittest
from llama_index.graph_stores.falkordb.base import FalkorDBGraphStore

# Set up Docker client
docker_client = docker.from_env()


class TestFalkorDBGraphStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method called once for the entire test class."""
        # Start FalkorDB container
        try:
            cls.container = docker_client.containers.run(
                "falkordb/falkordb:latest",
                detach=True,
                name="falkordb_test_instance",
                ports={"6379/tcp": 6379},
            )
            time.sleep(2)  # Allow time for the container to initialize
        except Exception as e:
            print(f"Error starting FalkorDB container: {e}")
            raise

        # Set up the FalkorDB Graph store
        cls.graph_store = FalkorDBGraphStore(url="redis://localhost:6379")

    @classmethod
    def tearDownClass(cls):
        """Teardown method called once after all tests are done."""
        try:
            cls.container.stop()
            cls.container.remove()
        except Exception as e:
            print(f"Error stopping/removing container: {e}")

    def test_base_graph(self):
        self.graph_store.upsert_triplet("node1", "related_to", "node2")

        # Check if the data has been inserted correctly
        result = self.graph_store.get("node1")
        expected_result = [
            "RELATED_TO",
            "node2",
        ]  # Expected data
        self.assertIn(expected_result, result)

        result = self.graph_store.get_rel_map(["node1"], 1)
        self.assertIn(expected_result, result["node1"])

        self.graph_store.delete("node1", "related_to", "node2")

        result = self.graph_store.get("node1")
        expected_result = []  # Expected data
        self.assertEqual(expected_result, result)

        self.graph_store.switch_graph("new_graph")
        self.graph_store.refresh_schema()


if __name__ == "__main__":
    unittest.main()
