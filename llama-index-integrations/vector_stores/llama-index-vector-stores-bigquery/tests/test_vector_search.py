import json
from unittest.mock import MagicMock
import pytest
from google.cloud import bigquery
from llama_index.core.schema import TextNode

from llama_index.vector_stores.bigquery import BigQueryVectorStore
from llama_index.vector_stores.bigquery.base import DistanceType
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilters,
    MetadataFilter,
    VectorStoreQueryResult,
)
from sql_assertions import assert_equivalent_sql_statements


def test_query_vector_search_generates_correct_sql_and_params(
    vector_store: BigQueryVectorStore,
):
    """It should construct and execute a VECTOR_SEARCH query with correct parameters."""
    # Given a VectorStoreQuery
    query = VectorStoreQuery(
        similarity_top_k=5,
        query_embedding=[1.0, 2.0, 3.0],
    )

    # When `query` is called
    vector_store.query(query)

    # Then it should call BigQuery with the correct query parameters
    vector_store.client.query_and_wait.assert_called_once()
    args, kwargs = vector_store.client.query_and_wait.call_args
    actual_query = args[0]
    job_config = kwargs["job_config"]

    expected_query_params = [
        bigquery.ScalarQueryParameter("top_k", "INTEGER", 5),
        bigquery.ScalarQueryParameter("distance_type", "STRING", "EUCLIDEAN"),
    ]
    assert isinstance(job_config, bigquery.QueryJobConfig)
    assert job_config.query_parameters == expected_query_params

    # And the actual SQL query should match the expected SQL query
    expected_query = f"""
    SELECT  base.node_id   AS node_id,
            base.text      AS text,
            base.metadata  AS metadata,
            base.embedding AS embedding,
            distance
    FROM   VECTOR_SEARCH(
            (
              SELECT node_id,
                     text,
                     metadata,
                     embedding
              FROM   `mock-project.mock_dataset.mock_table` ), 'embedding',
            ( SELECT [1.0, 2.0, 3.0] AS input_embedding ), 'input_embedding',
            top_k => @top_k,
            distance_type => @distance_type
    );
    """

    assert_equivalent_sql_statements(actual_query, expected_query)


def test_query_vector_store_with_filters_generates_correct_sql_and_params(
    vector_store: BigQueryVectorStore,
):
    """It should apply metadata filters and node ID constraints in the VECTOR_SEARCH query."""
    # Given a VectorStoreQuery
    query = VectorStoreQuery(
        similarity_top_k=5,
        query_embedding=[1.0, 2.0, 3.0],
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="author", value="ceo@company.com"),
                MetadataFilter(key="author", value="cfo@company.com"),
            ],
            condition="or",
        ),
        node_ids=["node1", "node2"],
    )

    # When `query` is called
    vector_store.query(query)

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
        bigquery.ScalarQueryParameter("top_k", "INTEGER", 5),
        bigquery.ScalarQueryParameter("distance_type", "STRING", "EUCLIDEAN"),
    ]
    assert isinstance(job_config, bigquery.QueryJobConfig)
    assert job_config.query_parameters == expected_query_params

    # And the actual SQL query should match the expected SQL query
    expected_query = f"""
     SELECT  base.node_id   AS node_id,
                base.text      AS text,
                base.metadata  AS metadata,
                base.embedding AS embedding,
                distance
    FROM
        VECTOR_SEARCH(
        (
            SELECT  node_id,
                    text,
                    metadata,
                    embedding
            FROM    `mock-project.mock_dataset.mock_table`
            WHERE   (SAFE.JSON_VALUE(metadata, '$."author"') = ? OR SAFE.JSON_VALUE(metadata, '$."author"') = ?)
                    AND node_id IN UNNEST(@node_ids)
        ), 'embedding',
            (SELECT [1.0, 2.0, 3.0] AS input_embedding),
            'input_embedding',
            top_k => @top_k,
            distance_type => @distance_type
    );
    """

    assert_equivalent_sql_statements(actual_query, expected_query)


@pytest.mark.parametrize(
    ("distance_type", "distance", "expected_similarity"),
    [
        (DistanceType.EUCLIDEAN, 1.5, 0.4),
        (DistanceType.COSINE, 0.9974149030430577, 0.9987074515215288),
        (DistanceType.DOT_PRODUCT, 17.0, 9.0),
    ],
)
def test_query_vector_store_result(
    mock_bigquery_client,
    distance_type: DistanceType,
    distance: float,
    expected_similarity: float,
):
    """It should return a VectorStoreQueryResult with correct nodes, IDs, and similarities based on distance type."""
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
    mock_row.distance = distance

    # Given a vector store
    vector_store = BigQueryVectorStore(
        project_id="mock-project",
        dataset_id="mock_dataset",
        table_id="mock_table",
        distance_type=distance_type,
        bigquery_client=mock_bigquery_client,
    )
    vector_store.client.query_and_wait.return_value = [mock_row]

    # And a VectorStoreQuery
    query = VectorStoreQuery(
        similarity_top_k=1,
        query_embedding=[1.5, 2.5, 3.5],
    )

    # When `query` is called
    results = vector_store.query(query)

    # Then the correct VectorStoreQueryResult should be returned
    expected_results = VectorStoreQueryResult(
        nodes=[
            TextNode(
                id_="node1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"author": "ceo@company.com"},
                text="Lorem Ipsum",
            )
        ],
        similarities=[expected_similarity],
        ids=["node1"],
    )

    assert results == expected_results
