LlamaIndex Readers Integration: Trino
Overview
The Trino Reader is designed to load data from any source accessible via a Trino cluster. Trino is a high-performance distributed SQL query engine that can query data across various sources (like data lakes, relational databases, etc.).

It retrieves documents by executing a user-provided SQL query against the specified Trino cluster using standard connection parameters. Each resulting row is converted into a LlamaIndex Document.

Installation
You can install the Trino Reader via pip:

Bash

pip install llama-index-readers-trino
Usage
Python

from llama_index.readers.trino import TrinoReader
from llama_index.core import VectorStoreIndex

# Initialize TrinoReader
reader = TrinoReader(
    host="trino.example.com",  # Trino host address
    port=8080,  # Trino port (default: 8080)
    user="your_user",  # User for connection
    catalog="hive",  # Target catalog (e.g., 'hive', 'iceberg')
    schema="default"  # Target schema
    # Add other parameters like http_scheme, auth, etc., as needed for your setup
)

# Define the SQL query
sql_query = """
    SELECT 
        product_id, 
        name, 
        description 
    FROM 
        hive.retail_data.products 
    WHERE 
        category = 'electronics'
"""

# Load data from Trino
print("Executing query on Trino and converting results to Documents...")
documents = reader.load_data(query=sql_query)

# Use the documents to build an index
index = VectorStoreIndex.from_documents(documents)

# Example: Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What are the names of the products returned from the query?")

print(response)
Implementation for the Trino reader will be found on LlamaHub and subsequently linked here.

This loader is designed to be used as a way to load data into
LlamaIndex and/or subsequently
used as a Tool in a LangChain Agent.