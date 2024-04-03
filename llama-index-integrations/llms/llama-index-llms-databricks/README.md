# LlamaIndex Llms Integration: Databricks

## Overview

Integrate with DataBricks LLMs APIs.

## Installation

```bash
pip install llama-index-llms-databricks
```

## Example

With environmental variables.

```.env
DATABRICKS_API_KEY=your_api_key
DATABRICKS_API_BASE=https://[your-work-space].cloud.databricks.com/serving-endpoints/[your-serving-endpoint]
```

```python
from llama_index.llms.databricks import DataBricks

# Initialize DataBricks LLM without explicitly passing the API key and base
llm = DataBricks(model="databricks-dbrx-instruct")

# Make a query to the LLM
response = llm.complete("Explain the importance of open source LLMs")

print(response)
```

Without environmental variables

```python
from llama_index.llms.databricks import DataBricks

# Set up the DataBricks class with the required model, API key and serving endpoint
llm = DataBricks(
    model="databricks-dbrx-instruct",
    api_key="your_api_key",
    api_base="https://[your-work-space].cloud.databricks.com/serving-endpoints/[your-serving-endpoint]",
)

# Call the complete method with a query
response = llm.complete("Explain the importance of open source LLMs")

print(response)
```
