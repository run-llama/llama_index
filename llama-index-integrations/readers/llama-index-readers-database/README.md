# LlamaIndex Readers Integration: Database

## Overview

Database Reader is a tool designed to query and load data from databases efficiently.

### Installation

You can install Database Reader via pip:

```bash
pip install llama-index-readers-database
```

## Usage

```python
from llama_index.core.schema import Document
from llama_index.readers.database import DatabaseReader

# Initialize DatabaseReader with the SQL database connection details
reader = DatabaseReader(
    sql_database="<SQLDatabase Object>",  # Optional: SQLDatabase object
    engine="<SQLAlchemy Engine Object>",  # Optional: SQLAlchemy Engine object
    uri="<Connection URI>",  # Optional: Connection URI
    scheme="<Scheme>",  # Optional: Scheme
    host="<Host>",  # Optional: Host
    port="<Port>",  # Optional: Port
    user="<Username>",  # Optional: Username
    password="<Password>",  # Optional: Password
    dbname="<Database Name>",  # Optional: Database Name
)

# Load data from the database using a query
documents = reader.load_data(
    query="<SQL Query>"  # SQL query parameter to filter tables and rows
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).
