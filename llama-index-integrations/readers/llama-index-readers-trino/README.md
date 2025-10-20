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

1. Code Completion & Hardening Roadmap 🛠️
This phase involves finishing the core features and ensuring the architecture is resilient.

Task 1: Complete the BaseReader Interface (Lazy Loading)
The current load_data works for small datasets but will crash on large ones. A production-ready reader must support streaming.

Action: Implement the lazy_load_data method required by the BaseReader interface.

Goal: Refactor the logic to use a Python generator (yield) inside this method. Instead of calling cur.fetchall() (which loads everything into memory), the method should fetch rows from the cursor one by one and yield a Document object for each row. This makes the reader memory-efficient.

Skill Highlight: Mastery of Python generators and streaming I/O to prevent Out-of-Memory (OOM) errors.

Task 2: Implement Robust Connection Management
The current TrinoReader relies on instance state (self._conn, self._cursor), which is vulnerable to being left open if a query fails mid-execution.

Action: Refactor the public load_data and lazy_load_data methods to use a context manager pattern (try...finally or a proper Python context manager).

Goal: Guarantee that the cursor and the connection are closed immediately after the query finishes or fails, preventing resource exhaustion on the Trino cluster.

Skill Highlight: SRE mindset and disciplined resource management.

Task 3: Enhance Metadata & Traceability
The current metadata ('row_id', 'source') is sparse. For advanced RAG, users need rich context.

Action: Augment the metadata dictionary creation within the load_data loop.

Goal: Include essential Trino metadata: the full query text used to retrieve the data, the timestamp of retrieval (for data freshness/auditability), and explicitly map the user-defined connection params (self.host, self.catalog, self.schema) into the metadata dictionary.

Skill Highlight: Understanding data governance and audit trails in ETL/ELT pipelines.

2. PR Preparation & Documentation Roadmap 📄
This phase focuses on packaging your contribution to meet LlamaIndex's high standards, making it easy for maintainers to review and approve.

Task 4: Develop Comprehensive Unit Tests
LlamaIndex requires at least 50% test coverage. You cannot rely on a live Trino cluster for tests.

Action: Create a tests/ directory and use unittest.mock to isolate your database logic.

Test 1 (Connection Failure): Mock the trino.dbapi.connect call to raise a DatabaseError and ensure your load_data function handles the exception gracefully (returning [] and logging an error, but not crashing).

Test 2 (Successful Load): Mock the cursor.fetchall() call to return a fixed set of sample data and verify that load_data correctly converts this raw data into the expected list of Document objects with rich metadata.

Test 3 (Lazy Load): Write a test that iterates over the generator returned by lazy_load_data and confirms the memory-efficient streaming behavior.

Skill Highlight: Test-Driven Development (TDD) and defensive mocking practices.

Task 5: Final Documentation and Issue Setup
Your contribution needs a clear narrative explaining its value.

Action: Draft the official Pull Request (PR) description and open a corresponding GitHub Issue.

PR Description: Clearly state the problem (e.g., "Generic SQL loaders lack Trino-specific session control") and how your TrinoReader solves it (native trino-python-client integration, resource safety).

Documentation File: Create a simple usage example file (e.g., usage_example.py) demonstrating how easily a user can instantiate the TrinoReader and pass a simple SQL query.

By following this roadmap, you will have a resilient, well-tested, and professionally documented component ready for contribution.