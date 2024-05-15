# Database Tool

This tool connects to a database (using SQLAlchemy under the hood) and allows an Agent to query the database and get information about the tables.

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/database.ipynb) and [here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/intro_to_tools.ipynb)

Here's an example usage of the DatabaseToolSpec.

```python
from llama_index.tools.database import DatabaseToolSpec
from llama_index.agent.openai import OpenAIAgent

db_tools = DatabaseToolSpec(
    scheme="postgresql",  # Database Scheme
    host="localhost",  # Database Host
    port="5432",  # Database Port
    user="postgres",  # Database User
    password="FakeExamplePassword",  # Database Password
    dbname="postgres",  # Database Name
)
agent = OpenAIAgent.from_tools(db_tools.to_tool_list())

agent.chat("What tables does this database contain")
agent.chat("Describe the first table")
agent.chat("Retrieve the first row of that table")
```

The tools available are:

`list_tables`: A tool to list the tables in the database schema
`describe_tables`: A tool to describe the schema of a table
`load_data`: A tool that accepts an SQL query and returns the result

This loader is designed to be used as a way to load data as a Tool in a Agent.
