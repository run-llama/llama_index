import json
from unittest.mock import MagicMock, patch

import pytest
from google.cloud import bigquery
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from llama_index.vector_stores.bigquery import BigQueryVectorStore
from sql_assertions import assert_equivalent_sql_statements


def test_get_nodes_with_filters_generates_correct_sql_and_params(
    vector_store: BigQueryVectorStore,
):
    """It should execute a parameterized query to get nodes based on filter criteria"""
    # Given filtering criteria
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="author", value="ceo@company.com"),
            MetadataFilter(key="author", value="cfo@company.com"),
        ],
        condition="or",
    )
    node_ids = ["node1", "node2"]

    # When `get_nodes` is called with the filtering criteria
    vector_store.get_nodes(node_ids, filters)

    # Then it should call BigQuery with the correct query parameters
    vector_store.client.query_and_wait.assert_called_once()
    args, kwargs = vector_store.client.query_and_wait.call_args
    actual_query = args[0]
    job_config = kwargs["job_config"]

    expected_query_params = [
        bigquery.ScalarQueryParameter(
            name=None, type_="STRING", value="ceo@company.com"
        ),
        bigquery.ScalarQueryParameter(
            name=None, type_="STRING", value="cfo@company.com"
        ),
        bigquery.ArrayQueryParameter(
            name="node_ids", array_type="STRING", values=["node1", "node2"]
        ),
    ]
    assert isinstance(job_config, bigquery.QueryJobConfig)
    assert job_config.query_parameters == expected_query_params

    # And the actual SQL query should match the expected SQL query
    expected_query = """
    SELECT  node_id,
            text,
            embedding,
            metadata
    FROM    `mock-project.mock_dataset.mock_table`
    WHERE   (SAFE.JSON_VALUE(metadata, '$."author"') = ? OR SAFE.JSON_VALUE(metadata, '$."author"') = ?)
            AND node_id IN UNNEST(@node_ids);
    """
    assert_equivalent_sql_statements(actual_query, expected_query)


def test_get_nodes_constructs_nodes_from_valid_metadata_row(
    vector_store: BigQueryVectorStore,
):
    """It should construct a Node when metadata includes valid `_node_content` and `_node_type`."""
    # Mock BigQuery returned record
    mock_row = MagicMock()
    mock_row.node_id = "node1"
    mock_row.text = "Lorem Ipsum"
    mock_row.embedding = [0.1, 0.2, 0.3]
    mock_row.metadata = {
        "author": "ceo@company.com",
        "_node_content": json.dumps(
            {
                "id_": "node1",
                "embedding": None,
                "metadata": {"author": "ceo@company.com"},
                "excluded_embed_metadata_keys": [],
                "excluded_llm_metadata_keys": [],
                "relationships": {},
                "metadata_template": "{key}: {value}",
                "metadata_separator": "\n",
                "text": "",
                "mimetype": "text/plain",
                "start_char_idx": None,
                "end_char_idx": None,
                "metadata_seperator": "\n",
                "text_template": "{metadata_str}\n\n{content}",
                "class_name": "TextNode",
            }
        ),
        "_node_type": "TextNode",
        "document_id": "None",
        "doc_id": "None",
        "ref_doc_id": "None",
    }
    vector_store.client.query_and_wait.return_value = [mock_row]

    # When `get_nodes` is called
    nodes = vector_store.get_nodes(node_ids=["node1"])

    # Then a node should be returned corresponding to the record returned from BigQuery
    assert nodes == [
        TextNode(
            id_="node1",
            text="Lorem Ipsum",
            embedding=[0.1, 0.2, 0.3],
            metadata={"author": "ceo@company.com"},
        )
    ]


@patch(
    "llama_index.vector_stores.bigquery.base.metadata_dict_to_node",
    side_effect=ValueError("_node_content not found in metadata dict."),
)
def test_get_nodes_falls_back_to_manual_textnode_on_metadata_parse_error(
    mock_metadata_dict_to_node, vector_store: BigQueryVectorStore
):
    """It should fall back to constructing a TextNode when metadata lacks `_node_content` and `_node_type`."""
    # Mock BigQuery returned record
    mock_row = MagicMock()
    mock_row.node_id = "node1"
    mock_row.text = "This is a test node"
    mock_row.embedding = [0.1, 0.2, 0.3]
    mock_row.metadata = {"author": "ceo@company.com"}
    vector_store.client.query_and_wait.return_value = [mock_row]

    # When `get_nodes` is called and the parser raises an Exception
    nodes = vector_store.get_nodes(node_ids=["node1"])
    mock_metadata_dict_to_node.assert_called_once_with({"author": "ceo@company.com"})
    assert mock_metadata_dict_to_node.raises_exception

    # Then a fallback TextNode is constructed and returned
    assert nodes == [
        TextNode(
            id_="node1",
            text="This is a test node",
            embedding=[0.1, 0.2, 0.3],
            metadata={"author": "ceo@company.com"},
        )
    ]


def test_get_nodes_without_arguments_raises_value_error(
    vector_store: BigQueryVectorStore,
):
    """It should raise a ValueError when neither node_ids nor filters is provided."""
    # When `get_nodes` is called without arguments, it should raise a `ValueError`
    with pytest.raises(ValueError):
        vector_store.get_nodes()
