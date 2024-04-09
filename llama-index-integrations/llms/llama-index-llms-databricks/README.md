# LlamaIndex Llms Integration: Databricks

## Overview

Integrate with Databricks LLMs APIs.

## Installation

```bash
pip install llama-index-llms-databricks
```

## Example

With environmental variables.

```.env
DATABRICKS_TOKEN=your_api_key
DATABRICKS_SERVING_ENDPOINT=https://[your-work-space].cloud.databricks.com/serving-endpoints
```

```python
from llama_index.llms.databricks import Databricks

# Initialize Databricks LLM without explicitly passing the API key and base
llm = Databricks(model="databricks-dbrx-instruct")

# Make a query to the LLM
response = llm.complete("Explain the importance of open source LLMs")

print(response)
```

Without environmental variables

```python
from llama_index.llms.databricks import Databricks

# Set up the Databricks class with the required model, API key and serving endpoint
llm = Databricks(
    model="databricks-dbrx-instruct",
    api_key="your_api_key",
    api_base="https://[your-work-space].cloud.databricks.com/serving-endpoints",
)

# Call the complete method with a query
response = llm.complete("Explain the importance of open source LLMs")

print(response)
```
