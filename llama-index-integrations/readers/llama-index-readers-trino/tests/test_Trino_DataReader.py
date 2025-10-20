import pytest
from unittest import mock
from llama_index.core.readers.base import BaseReader
import inspect 
from llama_index.base import TrinoReader

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
    mock_cursor.fetchall.return_value = MOCK_ROWS  # The rows to be yielded

    # 1. Initialize the reader
    reader = TrinoReader(
        host="mock-host", 
        port=8080, 
        user="test_user", 
        catalog="mock_catalog",
        schema="mock_schema"
    )
    
    query = "SELECT product_id, name, price FROM mock_table"

    # 2. Execute load_data
    documents = reader.load_data(query=query)

    # 3. Assertions
    # Check that the Trino connection was opened
    mock_trino_connect.assert_called_once_with(
        host="mock-host", port=8080, user="test_user", catalog="mock_catalog", schema="mock_schema"
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
        'source': 'raw_data',
        'catalog': 'mock_catalog',
        'schema' : 'mock_schema',
        'row_id': 0
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

    reader = TrinoReader(host="bad-host", port=1, user="bad_user", catalog="bad_catalog", schema="bad_schema")
    
    with pytest.raises(Exception) as excinfo:
        reader.load_data(query="SELECT * FROM dual")
    
    # Verify the specific error message is passed up
    assert "Trino connection failed" in str(excinfo.value)


@mock.patch("trino.dbapi.connect")
def test_trino_reader_handles_no_results(mock_trino_connect):
    """Tests that load_data returns an empty list when query returns no rows."""

    mock_cursor = mock_trino_connect.return_value.cursor.return_value
    mock_cursor.description = MOCK_COLUMNS
    mock_cursor.fetchall.return_value = [] # Empty rows list

    reader = TrinoReader(host="mock-host", port=8080, user="test_user", catalog="mock_catalog", schema="mock_schema")
    
    query = "SELECT * FROM empty_table WHERE 1=0"

    documents = reader.load_data(query=query)

    # Assert that no documents were returned
    assert len(documents) == 0
    assert isinstance(documents, list)

def test_trino_reader_implements_base_reader():
    """
    Asserts that the TrinoReader correctly inherits from LlamaIndex's BaseReader.
    This ensures the class is pluggable into the LlamaIndex framework.
    """
    
    # 1. Use the Method Resolution Order (__mro__) to get the full inheritance chain.
    #    This returns a tuple of classes: (TrinoReader, BaseReader, object, ...)
    inheritance_chain = TrinoReader.__mro__

    # 2. Extract the names of all base classes.
    names_of_base_classes = [cls.__name__ for cls in inheritance_chain]

    # 3. Assert that the required interface name exists in the inheritance chain.
    assert BaseReader.__name__ in names_of_base_classes, \
        "TrinoReader must inherit from llama_index.core.readers.base.BaseReader"
    
    # Optional: Verify the critical methods are defined (ensuring the implementation contract is met)
    # The BaseReader requires load_data (or lazy_load_data).
    assert inspect.isfunction(getattr(TrinoReader, 'load_data')), \
        "TrinoReader must define the public method 'load_data'."