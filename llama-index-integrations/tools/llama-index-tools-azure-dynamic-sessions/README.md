# Azure Dynamic Sessions Tool

This tool leverages Azure Dynamic Sessions Pool to enable an Agent to run generated Python code in a secure environment with very low latency.

In order to utilize the tool, you will need to have the Session Pool management endpoint first. More details in here: "some-url"

## Usage

A more detailed sample is located in a Jupyter notebook [here](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools/llama-index-tools-azure-dynamic-sessions/examples/azure_dynamic_sessions.ipynb)

Here's an example usage of the `AzureDynamicSessionsToolSpec`.

```python
from llama_index.tools.azure_dynamic_sessions import AzureDynamicSessionsToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.azure_openai import AzureOpenAI

llm = AzureOpenAI(
    model="gpt-35-turbo",
    deployment_name="gpt-35-deploy",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

dynamic_session_tool = AzureDynamicSessionsToolSpec(
    pool_managment_endpoint="your-pool-management-endpoint"
)

agent = ReActAgent.from_tools(dynamic_session_tool.to_tool_list(), llm=llm, verbose=True)

print(agent.chat("Tell me the current time in Seattle."))

print(dynamic_session_tool.code_interpreter("1+1"))
```

`code_interpreter`: Send a Python code to be executed in Azure Container Apps Dynamic Sessions and return the output in a JSON format.

`list_files`: List the files available in a Session under the path `/mnt/data`.

`upload_file`: Upload a file or a stream of data into a Session under the path `/mnt/data`.

`download_file`: Download a file by its path relative to the path `/mnt/data` to the tool's hosting agent.

This loader is designed to be used as a way to load data as a Tool in a Agent.
