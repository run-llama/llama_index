# Cassandra Database Tools

## Overview

The Cassandra Database Tools project is designed to help AI engineers efficiently integrate Large Language Models (LLMs) with Apache Cassandra® data. It facilitates optimized and safe interactions with Cassandra databases, supporting various deployments like Apache Cassandra®, DataStax Enterprise™, and DataStax Astra™.

## Key Features

- **Fast Data Access:** Optimized queries ensure most operations complete in milliseconds.
- **Schema Introspection:** Enhances the reasoning capabilities of LLMs by providing detailed schema information.
- **Compatibility:** Supports various Cassandra deployments, ensuring wide applicability.
- **Safety Measures:** Limits operations to SELECT queries and schema introspection to prioritize data integrity.

## Installation

Ensure your system has Python installed and proceed with the following installations via pip:

```bash
pip install python-dotenv cassio llama-index-tools-cassandra
```

Create a `.env` file for environmental variables related to Cassandra and Astra configurations, following the example structure provided in the notebook.

## Environment Setup

- For Cassandra: Configure `CASSANDRA_CONTACT_POINTS`, `CASSANDRA_USERNAME`, `CASSANDRA_PASSWORD`, and `CASSANDRA_KEYSPACE`.
- For DataStax Astra: Set `ASTRA_DB_APPLICATION_TOKEN`, `ASTRA_DB_DATABASE_ID`, and `ASTRA_DB_KEYSPACE`.

## How It Works

The toolkit leverages the Cassandra Query Language (CQL) and integrates with LLMs to provide an efficient query path determination for the user's requests, ensuring best practices for querying are followed. Using functions, the LLMs decision making can invoke the tool instead of designing custom queries. The result is faster and efficient access to Cassandra data for agents.

## Tools Included

- **`cassandra_db_schema`**: Fetches schema information, essential for the agent’s operation.
- **`cassandra_db_select_table_data`**: Allows selection of data from a specific keyspace and table.
- **`cassandra_db_query`**: An experimental tool that accepts fully formed query strings from the agent.

## Example Usage

Initialize the CassandraDatabase and set up the agent with the tools provided. Query the database by interacting with the agent as shown in the example [notebook](https://docs.llamaindex.ai/en/stable/examples/tools/cassandra/).
