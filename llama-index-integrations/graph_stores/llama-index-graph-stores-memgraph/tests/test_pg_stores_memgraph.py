import unittest
from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
)
from llama_index.core.schema import TextNode


class TestMemgraphGraphStore(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.pg_store = MemgraphPropertyGraphStore(
            username="", password="", url="bolt://localhost:7687"
        )

    def test_connection(self):
        """Test if connection to Memgraph is working."""
        try:
            self.pg_store.client.verify_connectivity()
            connected = True
        except Exception as e:
            connected = False
        self.assertTrue(connected, "Could not connect to Memgraph")

    def test_memgraph_pg_store(self):
        # Clear the database
        self.pg_store.structured_query("STORAGE MODE IN_MEMORY_ANALYTICAL")
        self.pg_store.structured_query("DROP GRAPH")
        self.pg_store.structured_query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")

        # Test upsert nodes
        entity1 = EntityNode(label="PERSON", name="Logan", properties={"age": 28})
        entity2 = EntityNode(label="ORGANIZATION", name="LlamaIndex")
        self.pg_store.upsert_nodes([entity1, entity2])

        # Assert the nodes are inserted correctly
        kg_nodes = self.pg_store.get(ids=[entity1.id])

        self.assertEqual(len(kg_nodes), 1)
        self.assertEqual(kg_nodes[0].name, "Logan")

        # Test inserting relations into Memgraph.
        relation = Relation(
            label="WORKS_FOR",
            source_id=entity1.id,
            target_id=entity2.id,
            properties={"since": 2023},
        )

        self.pg_store.upsert_relations([relation])

        # Assert the relation is inserted correctly by retrieving the relation map
        kg_nodes = self.pg_store.get(ids=[entity1.id])
        paths = self.pg_store.get_rel_map(kg_nodes, depth=1)
        self.assertEqual(len(paths), 1)
        path = paths[0]
        self.assertEqual(path[0].id, entity1.id)
        self.assertEqual(path[2].id, entity2.id)
        self.assertEqual(path[1].label, "WORKS_FOR")

        # Test inserting a source text node and 'MENTIONS' relations.
        source_node = TextNode(text="Logan (age 28), works for LlamaIndex since 2023.")

        relations = [
            Relation(
                label="MENTIONS", target_id=entity1.id, source_id=source_node.node_id
            ),
            Relation(
                label="MENTIONS", target_id=entity2.id, source_id=source_node.node_id
            ),
        ]

        self.pg_store.upsert_llama_nodes([source_node])
        self.pg_store.upsert_relations(relations)

        # Assert the source node and relations are inserted correctly
        llama_nodes = self.pg_store.get_llama_nodes([source_node.node_id])
        self.assertEqual(len(llama_nodes), 1)
        self.assertEqual(llama_nodes[0].text, source_node.text)

        # Test retrieving nodes by properties.
        kg_nodes = self.pg_store.get(properties={"age": 28})
        self.assertEqual(len(kg_nodes), 1)
        self.assertEqual(kg_nodes[0].label, "PERSON")
        self.assertEqual(kg_nodes[0].name, "Logan")

        # Test executing a structured query in Memgraph.
        query = "MATCH (n:`__Entity__`) RETURN n"
        result = self.pg_store.structured_query(query)
        self.assertEqual(len(result), 2)

        # Test upserting a new node with additional properties.
        new_node = EntityNode(
            label="PERSON", name="Logan", properties={"age": 28, "location": "Canada"}
        )
        self.pg_store.upsert_nodes([new_node])

        # Assert the node has been updated with the new property
        kg_nodes = self.pg_store.get(properties={"age": 28})
        self.assertEqual(len(kg_nodes), 1)
        self.assertEqual(kg_nodes[0].label, "PERSON")
        self.assertEqual(kg_nodes[0].name, "Logan")
        self.assertEqual(kg_nodes[0].properties["location"], "Canada")

        # Test deleting nodes from Memgraph.
        self.pg_store.delete(ids=[source_node.node_id])
        self.pg_store.delete(ids=[entity1.id, entity2.id])

        # Assert the nodes have been deleted
        nodes = self.pg_store.get(ids=[entity1.id, entity2.id])
        self.assertEqual(len(nodes), 0)
        text_nodes = self.pg_store.get_llama_nodes([source_node.node_id])
        self.assertEqual(len(text_nodes), 0)


if __name__ == "__main__":
    unittest.main()
