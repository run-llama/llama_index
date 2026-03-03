"""Unit tests for the apoc_sample parameter in Neo4j graph stores."""

from unittest.mock import MagicMock, patch

import neo4j
import pytest


def _get_schema_calls(mock_driver):
    """Extract execute_query calls that contain apoc.meta.data queries."""
    calls = mock_driver.execute_query.call_args_list
    schema_calls = []
    for c in calls:
        # First positional arg is neo4j.Query object
        query_obj = c.args[0] if c.args else None
        if isinstance(query_obj, neo4j.Query) and "apoc.meta.data" in query_obj.text:
            schema_calls.append(c)
    return schema_calls


@pytest.fixture()
def mock_neo4j_driver():
    """Create a mock Neo4j driver that passes connectivity checks."""
    with patch("neo4j.GraphDatabase.driver") as mock_driver_cls:
        driver_instance = MagicMock()
        mock_driver_cls.return_value = driver_instance

        # verify_connectivity succeeds
        driver_instance.__enter__ = MagicMock(return_value=driver_instance)
        driver_instance.__exit__ = MagicMock(return_value=False)

        # execute_query returns empty results
        driver_instance.execute_query.return_value = ([], None, None)

        # session mock for constraint creation
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        driver_instance.session.return_value = session

        yield driver_instance


class TestNeo4jGraphStoreApocSample:
    """Tests for apoc_sample parameter in Neo4jGraphStore."""

    def test_apoc_sample_default_empty_config(self, mock_neo4j_driver):
        """When apoc_sample is not provided, _apoc_meta_config should be empty."""
        from llama_index.graph_stores.neo4j import Neo4jGraphStore

        store = Neo4jGraphStore(
            username="neo4j",
            password="password",
            url="bolt://localhost:7687",
            refresh_schema=False,
        )
        assert store._apoc_meta_config == {}

    def test_apoc_sample_sets_config(self, mock_neo4j_driver):
        """When apoc_sample is provided, it should be stored in _apoc_meta_config."""
        from llama_index.graph_stores.neo4j import Neo4jGraphStore

        store = Neo4jGraphStore(
            username="neo4j",
            password="password",
            url="bolt://localhost:7687",
            refresh_schema=False,
            apoc_sample=1000,
        )
        assert store._apoc_meta_config == {"sample": 1000}

    def test_apoc_sample_zero_is_valid(self, mock_neo4j_driver):
        """apoc_sample=0 should produce {"sample": 0}, not be treated as falsy."""
        from llama_index.graph_stores.neo4j import Neo4jGraphStore

        store = Neo4jGraphStore(
            username="neo4j",
            password="password",
            url="bolt://localhost:7687",
            refresh_schema=False,
            apoc_sample=0,
        )
        assert store._apoc_meta_config == {"sample": 0}

    def test_refresh_schema_passes_config(self, mock_neo4j_driver):
        """refresh_schema should pass _apoc_meta_config to all three queries."""
        from llama_index.graph_stores.neo4j import Neo4jGraphStore

        store = Neo4jGraphStore(
            username="neo4j",
            password="password",
            url="bolt://localhost:7687",
            refresh_schema=False,
            apoc_sample=500,
        )

        store.refresh_schema()

        schema_calls = _get_schema_calls(mock_neo4j_driver)
        assert len(schema_calls) == 3

        for call in schema_calls:
            params = call.kwargs.get("parameters_", {})
            assert params.get("config") == {"sample": 500}, (
                f"Expected config with sample=500, got {params}"
            )

    def test_refresh_schema_passes_empty_config_by_default(self, mock_neo4j_driver):
        """When no apoc_sample is set, config should be an empty dict."""
        from llama_index.graph_stores.neo4j import Neo4jGraphStore

        store = Neo4jGraphStore(
            username="neo4j",
            password="password",
            url="bolt://localhost:7687",
            refresh_schema=False,
        )

        store.refresh_schema()

        schema_calls = _get_schema_calls(mock_neo4j_driver)
        assert len(schema_calls) == 3

        for call in schema_calls:
            params = call.kwargs.get("parameters_", {})
            assert params.get("config") == {}, f"Expected empty config, got {params}"


class TestNeo4jPropertyGraphStoreApocSample:
    """Tests for apoc_sample parameter in Neo4jPropertyGraphStore."""

    @pytest.fixture()
    def mock_pg_drivers(self):
        """Create mock drivers for property graph store."""
        with (
            patch("neo4j.GraphDatabase.driver") as mock_driver_cls,
            patch("neo4j.AsyncGraphDatabase.driver") as mock_async_driver_cls,
        ):
            driver_instance = MagicMock()
            mock_driver_cls.return_value = driver_instance

            async_driver_instance = MagicMock()
            mock_async_driver_cls.return_value = async_driver_instance

            # execute_query returns appropriate results per query
            def execute_query_side_effect(*args, **kwargs):
                query_obj = args[0] if args else None
                if isinstance(query_obj, neo4j.Query):
                    if "dbms.components" in query_obj.text:
                        # Return a mock version record for verify_version()
                        record = MagicMock()
                        record.data.return_value = {"versions": ["5.23.0"]}
                        return ([record], None, None)
                    if "apoc.meta.subGraph" in query_obj.text:
                        # Return empty nodes/rels for schema counts
                        record = MagicMock()
                        record.data.return_value = {
                            "nodes": [],
                            "relationships": [],
                        }
                        return ([record], None, None)
                return ([], None, None)

            driver_instance.execute_query.side_effect = execute_query_side_effect

            yield driver_instance

    def test_apoc_sample_default_empty_config(self, mock_pg_drivers):
        """When apoc_sample is not provided, _apoc_meta_config should be empty."""
        from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

        store = Neo4jPropertyGraphStore(
            username="neo4j",
            password="password",
            url="bolt://localhost:7687",
            refresh_schema=False,
        )
        assert store._apoc_meta_config == {}

    def test_apoc_sample_sets_config(self, mock_pg_drivers):
        """When apoc_sample is provided, it should be stored in _apoc_meta_config."""
        from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

        store = Neo4jPropertyGraphStore(
            username="neo4j",
            password="password",
            url="bolt://localhost:7687",
            refresh_schema=False,
            apoc_sample=2000,
        )
        assert store._apoc_meta_config == {"sample": 2000}

    def test_refresh_schema_passes_config(self, mock_pg_drivers):
        """refresh_schema should pass _apoc_meta_config in param_map."""
        from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

        store = Neo4jPropertyGraphStore(
            username="neo4j",
            password="password",
            url="bolt://localhost:7687",
            refresh_schema=False,
            apoc_sample=1000,
        )

        store.refresh_schema()

        schema_calls = _get_schema_calls(mock_pg_drivers)
        assert len(schema_calls) == 3

        for call in schema_calls:
            params = call.kwargs.get("parameters_", {})
            assert params.get("config") == {"sample": 1000}, (
                f"Expected config with sample=1000, got {params}"
            )
            # Verify EXCLUDED_LABELS is still passed alongside config
            assert "EXCLUDED_LABELS" in params, (
                "EXCLUDED_LABELS should still be in param_map"
            )
