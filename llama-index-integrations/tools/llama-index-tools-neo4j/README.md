# Neo4j Schema Query Builder

```bash
pip install llama-index-tools-neo4j
```

The `Neo4jQueryToolSpec` class provides a way to query a Neo4j graph database based on a provided schema definition. The class uses a language model to generate Cypher queries from user questions and has the capability to recover from Cypher syntax errors through a self-healing mechanism.

## Table of Contents

- [Usage](#usage)
  - [Initialization](#initialization)
  - [Running a Query](#running-a-query)
- [Features](#features)

## Usage

### Initialization

Initialize the `Neo4jQueryToolSpec` class with:

```python
from llama_index.tools.neo4j import Neo4jQueryToolSpec
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent

llm = OpenAI(model="gpt-4", openai_api_key="XXXX-XXXX", temperature=0)

gds_db = Neo4jQueryToolSpec(
    url="neo4j-url",
    user="neo4j-user",
    password="neo4j=password",
    llm=llm,
    database="neo4j",
)

tools = gds_db.to_tool_list()
agent = OpenAIAgent.from_tools(tools, verbose=True)
```

Where:

- `url`: Connection string for the Neo4j database.
- `user`: Username for the Neo4j database.
- `password`: Password for the Neo4j database.
- `llm`: A language model for generating Cypher queries (any type of LLM).
- `database`: The database name.

### Running a Query

To use the agent:

```python
# use agent
agent.chat("Where is JFK airport is located?")
```

```
Generated Cypher:

MATCH (p:Port {port_code: 'JFK'})
RETURN p.location_name_wo_diacritics AS Location

Final answer:
'The port code JFK is located in New York, United States.'
```

## Features

- **Schema-Based Querying**: The class extracts the Neo4j database schema to guide the Cypher query generation.
- **Self-Healing**: On a Cypher syntax error, the class corrects itself to produce a valid query.
- **Language Model Integration**: Uses a language model for natural and accurate Cypher query generation.
