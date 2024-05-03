# ChatGPT Plugin Tool

This tool allows Agents to load a plugin using a ChatGPT manifest file, and have the Agent interact with it.

## Usage

This tool has more extensive example usage documented in a Jupyter notebook [here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/chatgpt_plugin.ipynb)

```python
# Load the manifest
import requests
import yaml

f = requests.get(
    "https://raw.githubusercontent.com/sisbell/chatgpt-plugin-store/main/manifests/today-currency-converter.oiconma.repl.co.json"
).text
manifest = yaml.safe_load(f)

from llama_index.tools.chatgpt_plugin import ChatGPTPluginToolSpec
from llama_index.agent.openai import OpenAIAgent
from llama_index.tools.requests import RequestsToolSpec

requests_spec = RequestsToolSpec()
plugin_spec = ChatGPTPluginToolSpec(manifest)
# OR
plugin_spec = ChatGPTPluginToolSpec(
    manifest_url="https://raw.githubusercontent.com/sisbell/chatgpt-plugin-store/main/manifests/today-currency-converter.oiconma.repl.co.json"
)

agent = OpenAIAgent.from_tools(
    [*plugin_spec.to_tool_list(), *requests_spec.to_tool_list()], verbose=True
)
print(agent.chat("Convert 100 euros to CAD"))
```

`describe_plugin`: Describe the plugin that has been loaded.
`load_openapi_spec`: Returns the parsed OpenAPI spec that the class was initialized with

In addition to the above method, this tool makes all of the tools available from the OpenAPI Tool Spec and Requests Tool Spec available to the agent. The plugin OpenAPI definition is loaded into the OpenAPI tool spec, and authentication headers are passed in to the Requests tool spec

This loader is designed to be used as a way to load data as a Tool in a Agent.
