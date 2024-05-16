# Requests Tool

This tool provides the agent the ability to make HTTP requests. It can be combined with the OpenAPIToolSpec to interface with an OpenAPI server.

For security reasons, you must specify the hostname for the headers that you wish to provide. See [here for an example](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/openapi_and_requests.ipynb)

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/openapi_and_requests.ipynb)

Here's an example usage of the RequestsToolSpec.

```python
from llama_index.tools.requests import RequestsToolSpec
from llama_index.agent.openai import OpenAIAgent

domain_headers = {
    "api.openai.com": {
        "Authorization": "Bearer sk-your-key",
        "Content-Type": "application/json",
    }
}

tool_spec = RequestsToolSpec(domain_headers=domain_headers)

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("")
```

`get_request`: Performs a get request against the URL
`post_request`: Performs a post request against the URL
`patch_request`: Performs a patch request against the URL

This loader is designed to be used as a way to load data as a Tool in a Agent.
