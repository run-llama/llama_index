# Database Tool

This tool connects to a database (using SQLAlchemy under the hood) and allows an Agent to query the database and get information about the tables.

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-database/examples/database.ipynb).

Here's an example usage of the DatabaseToolSpec.

```python
from llama_index.tools.database import DatabaseToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

db_tools = DatabaseToolSpec(
    scheme="postgresql",  # Database Scheme
    host="localhost",  # Database Host
    port="5432",  # Database Port
    user="postgres",  # Database User
    password="FakeExamplePassword",  # Database Password
    dbname="postgres",  # Database Name
)
agent = FunctionAgent(
    tools=db_tools.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("What tables does this database contain"))
print(await agent.run("Describe the first table"))
print(await agent.run("Retrieve the first row of that table"))
```

The tools available are:

`list_tables`: A tool to list the tables in the database schema
`describe_tables`: A tool to describe the schema of a table
`load_data`: A tool that accepts an SQL query and returns the result

This loader is designed to be used as a way to load data as a Tool in a Agent.
