# LlamaIndex Llms Integration: Cortex

## Overview

Integrate with Snowflake Cortex API.
3 ways to authenticate:

1. Snowpark Session object (recommended way)
   this allows authentication via Snowpark Container Services default token, via Oauth,
   password, private key, web browser, or any other method.

   Guide to creating sessions: https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session

2. Path to a private key file. Encrypted private keys unsupported. For encrypted keys: use a Snowpark Session instead, with the 'private_key_file_pwd' parameter.

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
from llama_index.llms.cortex import Cortex

connection_parameters = {
    "account": "<your snowflake account>",
    "user": "<your snowflake user>",
    "role": "<your snowflake role>",
    "database": "<your snowflake database>",
    "schema": "<your snowflake schema",
    "warehouse": "<your snowflake warehouse>",
    "private_key_file": "<path to file>",
    "authenticator": "JWT_AUTHENTICATOR",  # use this for private key
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

## Connect in an SPCS environment

```python
# That's it! That's all we need.
llm = Cortex(model="llama3.2-1b")
completion_response = llm.complete(
    "write me a haiku about a snowflake", temperature=0.0
)
print(completion_response)
```

## Create a session within an SPCS environment

```python
import os
from snowflake.snowpark import Session
from llama_index.llms.cortex import Cortex
from llama_index.llms.cortex import utils as cortex_utils

#! Note now the user and role parameters are left blank for SPCS !
connection_parameters = {
    "account": "<your snowflake account>",
    "database": "<your snowflake database>",
    "schema": "<your snowflake schema",
    "warehouse": "<your snowflake warehouse>",
    "token": cortex_utils.get_default_spcs_token(),
    "host": cortex_utils.get_spcs_base_url(),
    "authenticator": "OAUTH",
}
session = Session.builder.configs(connection_parameters).create()

llm = Cortex(model="llama3.2-1b", session=session)

completion_response = llm.complete(
    "write me a haiku about a snowflake", temperature=0.0
)
print(completion_response)
```

## TODO

1 snowflake token counting
2 Pull metadata for snowflake models from Snowflake official documentation (support ticket is out, they said they'll get back to me 4-20-2025)
