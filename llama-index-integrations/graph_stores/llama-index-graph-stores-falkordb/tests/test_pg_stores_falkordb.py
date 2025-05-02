import time
import unittest
import docker
from llama_index.graph_stores.falkordb import FalkorDBPropertyGraphStore
from llama_index.core.graph_stores.types import Relation, EntityNode
from llama_index.core.schema import TextNode

# Set up Docker client
docker_client = docker.from_env()


class TestFalkorDBPropertyGraphStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method called once for the entire test class."""
        # Start FalkorDB container
        try:
            cls.container = docker_client.containers.run(
                "falkordb/falkordb:latest",
                detach=True,
                name="falkordb_test_instance_pg",
                ports={"6379/tcp": 6380},
            )
            time.sleep(2)  # Allow time for the container to initialize
        except Exception as e:
            print(f"Error starting FalkorDB container: {e}")
            raise

        # Set up the property graph store and clear database
        cls.pg_store = FalkorDBPropertyGraphStore(url="redis://localhost:6380")
        cls.pg_store.structured_query("MATCH (n) DETACH DELETE n")  # Clear the database

    @classmethod
    def tearDownClass(cls):
        """Teardown method called once after all tests are done."""
        try:
            cls.container.stop()
            cls.container.remove()
        except Exception as e:
            print(f"Error stopping/removing container: {e}")

    def test_pg_graph(self):
        # Create two entity nodes
        entity1 = EntityNode(label="PERSON", name="Logan", properties={"age": 28})
        entity2 = EntityNode(label="ORGANIZATION", name="LlamaIndex")

        # Create a relation
        relation = Relation(
            label="WORKS_FOR",
            source_id=entity1.id,
            target_id=entity2.id,
            properties={"since": 2023},
        )

        self.pg_store.upsert_nodes([entity1, entity2])
        self.pg_store.upsert_relations([relation])

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

        self.pg_store.upsert_llama_nodes([source_node])
        self.pg_store.upsert_relations(relations)

        kg_nodes = self.pg_store.get(ids=[entity1.id])
        self.assertEqual(len(kg_nodes), 1)
        self.assertEqual(kg_nodes[0].label, "PERSON")
        self.assertEqual(kg_nodes[0].name, "Logan")

        kg_nodes = self.pg_store.get(properties={"age": 28})
        self.assertEqual(len(kg_nodes), 1)
        self.assertEqual(kg_nodes[0].label, "PERSON")
        self.assertEqual(kg_nodes[0].name, "Logan")

        # Get paths from a node
        paths = self.pg_store.get_rel_map(kg_nodes, depth=1)
        for path in paths:
            self.assertEqual(path[0].id, entity1.id)
            self.assertEqual(path[2].id, entity2.id)
            self.assertEqual(path[1].id, relation.id)

        query = "MATCH (n:`__Entity__`) RETURN n"
        result = self.pg_store.structured_query(query)
        self.assertEqual(len(result), 2)

        # Get the original text node back
        llama_nodes = self.pg_store.get_llama_nodes([source_node.node_id])
        self.assertEqual(len(llama_nodes), 1)
        self.assertEqual(llama_nodes[0].text, source_node.text)

        # Upsert a new node
        new_node = EntityNode(
            label="PERSON", name="Logan", properties={"age": 28, "location": "Canada"}
        )
        self.pg_store.upsert_nodes([new_node])
        kg_nodes = self.pg_store.get(properties={"age": 28})
        self.assertEqual(len(kg_nodes), 1)
        self.assertEqual(kg_nodes[0].label, "PERSON")
        self.assertEqual(kg_nodes[0].name, "Logan")
        self.assertEqual(kg_nodes[0].properties["location"], "Canada")

        # Deleting
        # Delete our entities
        self.pg_store.delete(ids=[entity1.id, entity2.id])

        # Delete our text nodes
        self.pg_store.delete(ids=[source_node.node_id])

        nodes = self.pg_store.get(ids=[entity1.id, entity2.id])
        self.assertEqual(len(nodes), 0)

        text_nodes = self.pg_store.get_llama_nodes([source_node.node_id])
        self.assertEqual(len(text_nodes), 0)

        self.pg_store.switch_graph("new_graph")
        self.pg_store.refresh_schema()


if __name__ == "__main__":
    unittest.main()
