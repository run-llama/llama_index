# LlamaIndex Readers Integration: Database

## Overview

Database Reader is a tool designed to query and load data from databases efficiently.

### Key features

- Accepts connection via `SQLDatabase`, SQLAlchemy `Engine`, full URI, or discrete credentials
- Optional `schema` selection (namespace)
- Column-level metadata mapping (`metadata_cols`) and text exclusion (`excluded_text_cols`)
- Custom `id_` generation function (`document_id`)
- Supports streaming (`lazy_load_data`) and async (`aload_data`)

### Installation

You can install Database Reader via pip:

```bash
pip install llama-index-readers-database
```

## Usage

```python
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

```python
# Initialize DatabaseReader with the SQL connection string and custom database schema
from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    uri="postgresql+psycopg2://user:pass@localhost:5432/mydb",
    schema="warehouse",  # optional namespace
)
# Streaming variant, excluded id from text_resource
for doc in reader.lazy_load_data(
    query="SELECT * FROM warehouse.big_table", excluded_text_cols={"id"}
):
    process(doc)

# Async variant, added region to metadata
docs_async = await reader.aload_data(
    query="SELECT * FROM warehouse.big_table", metadata_cols=["region"]
)
```

```python
# Advanced usage with custom named metadata columns, columns excluded from the `Document.text_resource`, and a dynamic `Document.id_` generated from row data and a fstring template
from llama_index.readers.database import DatabaseReader

reader_media = DatabaseReader(
    uri="postgresql+psycopg2://user:pass@localhost:5432/mydb",
    schema="media",  # optional namespace
)

docs = reader_media.load_data(
    query="SELECT id, title, body, updated_at FROM media.articles",
    metadata_cols=[
        ("id", "article_id"),
        "updated_at",
    ],  # map / include in metadata
    excluded_text_cols=["updated_at"],  # omit from text
    document_id=lambda row: f"media-articles-{row['id']}",  # custom document id
)
```

This loader is designed to be used as a way to load data into
[LlamaIndex](https://github.com/run-llama/llama_index/tree/main/llama_index).
