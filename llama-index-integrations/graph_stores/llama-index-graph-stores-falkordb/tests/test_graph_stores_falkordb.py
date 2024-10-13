import pytest
from unittest.mock import MagicMock, patch
from llama_index.graph_stores.falkordb.base import FalkorDBGraphStore


class TestFalkorDBGraphStore:
    @pytest.fixture()
    def mock_falkordb(self):
        with patch("llama_index.graph_stores.falkordb.base.FalkorDB") as mock:
            yield mock

    def test_falkordbgraphstore_delete(self, mock_falkordb: MagicMock):
        """Test the `delete` method of FalkorDBGraphStore."""
        mock_graph = mock_falkordb.from_url.return_value.select_graph.return_value
        mock_graph.query.return_value = (
            MagicMock()
        )  # Adjusting this to simulate a query return

        graph_store = FalkorDBGraphStore("mock_url")
        graph_store.delete("subject_id", "related_to", "object_id")

        expected_delete_rel_query = """
            MATCH (n1:`Entity`)-[r:`RELATED_TO`]->(n2:`Entity`)
            WHERE n1.id = $subj AND n2.id = $obj DELETE r
        """

        mock_graph.query.assert_any_call(
            expected_delete_rel_query, params={"subj": "subject_id", "obj": "object_id"}
        )

    def test_falkordbgraphstore_get(self, mock_falkordb: MagicMock):
        """Test the `get` method of FalkorDBGraphStore."""
        mock_graph = mock_falkordb.from_url.return_value.select_graph.return_value
        mock_graph.query.return_value.result_set = [["type", "obj_id"]]

        graph_store = FalkorDBGraphStore("mock_url")
        result = graph_store.get("subject_id")

        mock_graph.query.assert_called_once_with(
            graph_store.get_query, params={"subj": "subject_id"}, read_only=True
        )

    def test_falkordbgraphstore_get_rel_map(self, mock_falkordb: MagicMock):
        """Test the `get_rel_map` method of FalkorDBGraphStore."""
        mock_graph = mock_falkordb.from_url.return_value.select_graph.return_value
        mock_graph.query.return_value = [
            [
                MagicMock(
                    nodes=lambda: [
                        MagicMock(properties={"id": "subj"}),
                        MagicMock(properties={"id": "obj"}),
                    ],
                    edges=lambda: [MagicMock(relation="REL")],
                )
            ]
        ]

        graph_store = FalkorDBGraphStore("mock_url")
        subjs = ["subj1", "subj2"]
        result = graph_store.get_rel_map(subjs=subjs, depth=2, limit=30)

        # You may want to add assertions here based on expected results

    def test_falkordbgraphstore_get_schema(self, mock_falkordb: MagicMock):
        """Test the `get_schema` method of FalkorDBGraphStore."""
        mock_graph = mock_falkordb.from_url.return_value.select_graph.return_value
        mock_graph.query.side_effect = [
            [["property1"], ["property2"]],
            [["rel1"], ["rel2"]],
        ]

        graph_store = FalkorDBGraphStore("mock_url")
        schema = graph_store.get_schema(refresh=True)

        # You may want to add assertions here based on expected results

    def test_falkordbgraphstore_query(self, mock_falkordb: MagicMock):
        """Test the `query` method."""
        mock_graph = mock_falkordb.from_url.return_value.select_graph.return_value
        mock_graph.query.return_value.result_set = [["result"]]

        graph_store = FalkorDBGraphStore("mock_url")
        result = graph_store.query("mock_query", params={"key": "value"})

        mock_graph.query.assert_called_once_with("mock_query", params={"key": "value"})

    def test_falkordbgraphstore_refresh_schema(self, mock_falkordb: MagicMock):
        """Test the `refresh_schema` method."""
        mock_graph = mock_falkordb.from_url.return_value.select_graph.return_value
        mock_graph.query.side_effect = [
            [["property1"], ["property2"]],
            [["rel1"], ["rel2"]],
        ]

        graph_store = FalkorDBGraphStore("mock_url")
        graph_store.refresh_schema()

        # You may want to add assertions here based on expected results

    def test_falkordbgraphstore_upsert_triplet(self, mock_falkordb: MagicMock):
        """Test the `upsert_triplet` method of FalkorDBGraphStore."""
        mock_graph = mock_falkordb.from_url.return_value.select_graph.return_value
        graph_store = FalkorDBGraphStore("mock_url")

        graph_store.upsert_triplet("subject_id", "related_to", "object_id")

        expected_query = """
            MERGE (n1:`Entity` {id:$subj})
            MERGE (n2:`Entity` {id:$obj})
            MERGE (n1)-[:`RELATED_TO`]->(n2)
        """

        mock_graph.query.assert_called_once_with(
            expected_query, params={"subj": "subject_id", "obj": "object_id"}
        )
