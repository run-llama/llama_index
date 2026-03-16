from unittest.mock import MagicMock, patch

import pytest
from google.cloud import bigquery

from llama_index.vector_stores.bigquery import BigQueryVectorStore

_RealBigQueryClient = bigquery.Client
_RealBigQueryDataset = bigquery.Dataset
_RealTable = bigquery.Table


@pytest.fixture
def mock_bigquery_client():
    with patch(
        "llama_index.vector_stores.bigquery.base.bigquery.Client"
    ) as mock_client_cls:
        mock_client = MagicMock(spec=_RealBigQueryClient)
        mock_client.project = "mock-project"

        def create_dataset(dataset_ref, exists_ok=False):
            mock_dataset = MagicMock(spec=_RealBigQueryDataset)
            mock_dataset.dataset_id = dataset_ref.dataset_id
            return mock_dataset

        mock_client.create_dataset.side_effect = create_dataset

        def create_table(table_obj, exists_ok=False):
            mock_table = MagicMock(spec=_RealTable)
            mock_table.schema = table_obj.schema
            mock_table.table_id = table_obj.table_id
            return mock_table

        mock_client.create_table.side_effect = create_table

        mock_client_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def vector_store(mock_bigquery_client) -> BigQueryVectorStore:
    return BigQueryVectorStore(
        dataset_id="mock_dataset",
        table_id="mock_table",
        bigquery_client=mock_bigquery_client,
    )
