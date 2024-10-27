import os
import time
import docker
import unittest
from llama_index.graph_stores.falkordb.base import FalkorDBGraphStore

# Set up FalkorDB URL
falkordb_url = os.getenv("FALKORDB_TEST_URL", "redis://localhost:6379")

# Set up Docker client
docker_client = docker.from_env()


@unittest.skipIf(falkordb_url is None, "No FalkorDB URL provided")
class TestFalkorDBGraphStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method called once for the entire test class."""
        # Stop and remove the container if it already exists
        try:
            existing_container = docker_client.containers.get("falkordb_test_instance")
            existing_container.stop()
            existing_container.remove()
        except docker.errors.NotFound:
            pass  # If no container exists, we can proceed

        # Start FalkorDB container
        cls.container = docker_client.containers.run(
            "falkordb/falkordb:latest",
            detach=True,
            name="falkordb_test_instance",
            ports={"6379/tcp": 6379},
        )
        time.sleep(2)  # Allow time for the container to initialize

        # Set up the FalkorDB store and clear database
        cls.graph_store = FalkorDBGraphStore(url=falkordb_url)
        cls.graph_store.structured_query(
            "MATCH (n) DETACH DELETE n"
        )  # Clear the database

    @classmethod
    def tearDownClass(cls):
        """Teardown method called once after all tests are done."""
        cls.container.stop()
        cls.container.remove()

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
