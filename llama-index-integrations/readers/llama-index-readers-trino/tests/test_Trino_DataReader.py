import pytest
from unittest import mock

# Assuming your TrinoReader is here:
from llama_index.readers.trino import TrinoReader 

# Mock Data to simulate a query result from Trino
MOCK_COLUMNS = [('product_id', 'varchar'), ('name', 'varchar'), ('price', 'double')]
MOCK_ROWS = [
    (101, 'Laptop Pro X', 1200.00),
    (102, 'Gaming Mouse', 45.99),
    (103, 'USB-C Hub', 25.00),
]

@mock.patch("trino.dbapi.connect")
def test_trino_reader_loads_data_successfully(mock_trino_connect):
    """Tests that load_data connects, queries, and returns correct Documents."""

    # Configure the mock connection and cursor object
    mock_cursor = mock_trino_connect.return_value.cursor.return_value
    mock_cursor.description = MOCK_COLUMNS
    mock_cursor.__iter__.return_value = MOCK_ROWS  # The rows to be yielded

    # 1. Initialize the reader
    reader = TrinoReader(
        host="mock-host", 
        port=8080, 
        user="test_user", 
        catalog="mock_catalog"
    )
    
    query = "SELECT product_id, name, price FROM mock_table"

    # 2. Execute load_data
    documents = reader.load_data(query=query)

    # 3. Assertions
    # Check that the Trino connection was opened
    mock_trino_connect.assert_called_once_with(
        host="mock-host", port=8080, user="test_user", catalog="mock_catalog"
    )
    # Check that the query was executed
    mock_cursor.execute.assert_called_once_with(query)
    
    # Check document count
    assert len(documents) == 3 

    # Check the first document's content (Row 101)
    doc1 = documents[0]
    
    # Check the document text (Concatenated row values)
    expected_text_1 = "product_id: 101, name: Laptop Pro X, price: 1200.0"
    assert doc1.text == expected_text_1
    
    # Check the document metadata (Structured column data)
    expected_metadata_1 = {
        "product_id": 101,
        "name": "Laptop Pro X",
        "price": 1200.00,
    }
    assert doc1.metadata == expected_metadata_1

    # Check the second document's content (Row 102)
    doc2 = documents[1]
    expected_text_2 = "product_id: 102, name: Gaming Mouse, price: 45.99"
    assert doc2.text == expected_text_2

@mock.patch("trino.dbapi.connect")
def test_trino_reader_handles_connection_error(mock_trino_connect):
    """Tests that a Trino connection error is handled."""

    # Simulate a connection error when 'connect' is called
    mock_trino_connect.side_effect = Exception("Trino connection failed")

    reader = TrinoReader(host="bad-host", port=1, user="bad_user", catalog="bad_catalog")
    
    with pytest.raises(Exception) as excinfo:
        reader.load_data(query="SELECT * FROM dual")
    
    # Verify the specific error message is passed up
    assert "Trino connection failed" in str(excinfo.value)


@mock.patch("trino.dbapi.connect")
def test_trino_reader_handles_no_results(mock_trino_connect):
    """Tests that load_data returns an empty list when query returns no rows."""

    mock_cursor = mock_trino_connect.return_value.cursor.return_value
    mock_cursor.description = MOCK_COLUMNS
    mock_cursor.__iter__.return_value = [] # Empty rows list

    reader = TrinoReader(host="mock-host", port=8080, user="test_user", catalog="mock_catalog")
    
    query = "SELECT * FROM empty_table WHERE 1=0"

    documents = reader.load_data(query=query)

    # Assert that no documents were returned
    assert len(documents) == 0
    assert isinstance(documents, list)