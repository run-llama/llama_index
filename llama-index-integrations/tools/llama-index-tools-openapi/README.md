# OpenAPI Tool

```bash
pip install llama-index-tools-openapi
```

This tool loads an OpenAPI spec and allow the Agent to retrieve endpoints and details about endpoints. The RequestsToolSpec can also be loaded into the agent to allow the agent to hit the necessary endpoints with a REST request.

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-openapi/examples/openapi_and_requests.ipynb)

Here's an example usage of the OpenAPIToolSpec.

```python
from llama_index.tools.openapi import OpenAPIToolSpec
from llama_index.agent.openai import OpenAIAgent

f = requests.get(
    "https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/openai.com/1.2.0/openapi.yaml"
).text
open_api_spec = yaml.safe_load(f)
# OR
open_spec = OpenAPIToolSpec(
    url="https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/openai.com/1.2.0/openapi.yaml"
)


tool_spec = OpenAPIToolSpec(open_api_spec)

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("What is the base url for the API")
agent.chat("What parameters does the x endpoint need?")
```

`load_openapi_spec`: Returns the parsed OpenAPI spec that the class was initialized with

This loader is designed to be used as a way to load data as a Tool in a Agent.
