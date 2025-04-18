# LlamaIndex Llms Integration: Cortex

## Overview

Integrate with Snowflake Cortex API.
3 ways to authenticate:

1. Snowpark Session object (recommended way)
   this allows authentication via Snowpark Container Services default token, via Oauth,
   password, private key, web browser, or any other method.
2. Path to a private key file
3. JWT token

## Installation

```bash
pip install llama-index-llms-cortex
```

## Example using a Private Key

```python
import os
from llama_index.llms.cortex import Cortex


llm = Cortex(
    model="llama3.2-1b",
    user=os.environ["YOUR_SF_USER"],
    account=os.environ["YOUR_SF_ACCOUNT"],
    private_key_file=os.environ["PATH_TO_SF_PRIVATE_KEY"],
)

completion_response = llm.complete(
    "write me a haiku about a snowflake", temperature=0.0
)
print(completion_response)
```

## Example Using a Session

```python
import os
from snowflake.snowpark import Session

connection_parameters = {
    "account": "<your snowflake account>",
    "user": "<your snowflake user>",
    "role": "<your snowflake role>",
    "database": "<your snowflake database>",
    "schema": "<your snowflake schema",
    "warehouse": "<your snowflake warehouse>",
    "authenticator": "externalbrowser",
}
session = Session.builder.configs(connection_parameters).create()


llm = Cortex(
    model="llama3.2-1b",
    user=os.environ["YOUR_SF_USER"],
    account=os.environ["YOUR_SF_ACCOUNT"],
    session=session,
)

completion_response = llm.complete(
    "write me a haiku about a snowflake", temperature=0.0
)
print(completion_response)
```
