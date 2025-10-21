TrinoReader is a custom Data Loader designed to solve the problem of robust data ingestion for LlamaiNdex RAG pipelines on a Trino Data lake

## Quick Start Guide

### Installation

Install the required Python packages, including the core reader and the native `trino` client:

```bash
pip install llama-index-core trino pandas
```

### Usage Example: Data Ingestion

The TrinoReader is instantiated with standard connection parameters and uses the load_data method to execute a query and retrieve documents ready for indexing.

```Python
import logging
from llama_index.core.schema import Document
from your_module import TrinoReader # Assumed import

# 1. Define the SQL Query (Explicitly list columns for best practice)
query_to_index = """
SELECT c_custkey, c_name, c_acctbal
FROM tpch.tiny.customer
WHERE c_nationkey = 1
LIMIT 5;
"""

# 2. Instantiate the Reader
trino_data_loader = TrinoReader(
    host="localhost",
    port=8080,
    user="rag_user",
    catalog="tpch",
    schema="tiny"
)

# 3. Execute the Ingestion
print(f"Executing query on Trino...")
try:
    documents: List[Document] = trino_data_loader.load_data(query=query_to_index)

    # 4. Verification: Inspect the RAG-ready Document
    if documents:
        print(f"Successfully loaded {len(documents)} documents.")
        print("\n--- Example Document (High-Density Context) ---")
        print(f"Text Content: {documents[0].text}")
        print(f"Metadata: {documents[0].metadata}")
        print("------------------------------------------")

except Exception as e:
    logging.error(f"FATAL: Data loading failed: {e}")

```

### Contributing

This is an open-source project. If you have any suggestions for improvement, or would like to contribute a fix, please feel free to submit a pull request.

### Focus Areas for Contribution:

Implementing the lazy_load_data generator for memory-efficient streaming of massive tables.

Adding support for advanced Trino authentication methods (Kerberos, JWT).
