# LlamaIndex Llms Integration: Cortex

## Overview

Integrate with Snowflake Cortex API.

## Installation

```bash
pip install llama-index-llms-cortex
```

## Example

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
