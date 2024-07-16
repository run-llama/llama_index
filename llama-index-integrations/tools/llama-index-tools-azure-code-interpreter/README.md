# Azure Code Interpreter Tool

This tool leverages Azure Dynamic Sessions Pool to enable an Agent to run generated Python code in a secure environment with very low latency.

In order to utilize the tool, you will need to have the Session Pool management endpoint first. [Learn more](https://aka.ms/aca/sessions)

## Prerequisites

- Make sure to create a Session Pool and note down the `poolManagementEndpoint`.

- In order to have the code execution right, the correct role needs to be assigned to the current user agent. Be sure to assign `Session Pool Executor` role to the correct user agent's identity (e.g. User Email, Service Principal, Managed Identity, etc.) in Session Pool's access control panel through the Portal or CLI. [Learn more](https://aka.ms/aca/sessions)

## Usage

A more detailed sample is located in a Jupyter notebook [here](https://github.com/run-llama/llama_index/tree/main/docs/docs/examples/tools/azure_code_interpreter.ipynb)

Here's an example usage of the `AzureCodeInterpreterToolSpec`.

1. First, install the Azure Dynamic Sessions package using `pip`:

```
pip install llama-index-tools-azure-code-interpreter
```

2. Create a file named `.env` in the same directory as your script with the following content:

```
AZURE_POOL_MANAGEMENT_ENDPOINT=<poolManagementEndpoint>
```

3. Next, set up the Dynamic Sessions tool and a LLM agent:

```python
from llama_index.tools.azure_code_interpreter import (
    AzureCodeInterpreterToolSpec,
)
from llama_index.core.agent import ReActAgent
from llama_index.llms.azure_openai import AzureOpenAI

llm = AzureOpenAI(
    model="gpt-35-turbo",
    deployment_name="gpt-35-deploy",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

code_interpreter_spec = AzureCodeInterpreterToolSpec(
    pool_management_endpoint=os.getenv("AZURE_POOL_MANAGEMENT_ENDPOINT")
)

agent = ReActAgent.from_tools(
    code_interpreter_spec.to_tool_list(), llm=llm, verbose=True
)
```

4. Use the tool as you need:

```python
print(agent.chat("Tell me the current time in Seattle."))

"""
Sample Return:
Thought: To provide the current time in Seattle, I need to calculate it based on the current UTC time and adjust for Seattle's time zone, which is Pacific Daylight Time (PDT) during daylight saving time and Pacific Standard Time (PST) outside of daylight saving time. PDT is UTC-7, and PST is UTC-8. I can use the code interpreter tool to get the current UTC time and adjust it accordingly.
Action: code_interpreter
Action Input: {'python_code': "from datetime import datetime, timedelta; import pytz; utc_now = datetime.now(pytz.utc); seattle_time = utc_now.astimezone(pytz.timezone('America/Los_Angeles')); seattle_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')"}
Observation: {'$id': '1', 'status': 'Success', 'stdout': '', 'stderr': '', 'result': '2024-05-04 13:54:09 PDT-0700', 'executionTimeInMilliseconds': 120}
Thought: I can answer without using any more tools. I'll use the user's language to answer.
Answer: The current time in Seattle is 2024-05-04 13:54:09 PDT.
The current time in Seattle is 2024-05-04 13:54:09 PDT.
"""

print(dynamic_session_tool.code_interpreter("1+1"))

"""
Sample Return:
{'$id': '1', 'status': 'Success', 'stdout': '', 'stderr': '', 'result': 2, 'executionTimeInMilliseconds': 11}
"""
```

## Included Tools

The `AzureCodeInterpreterToolSpec` provides the following tools to the agent:

`code_interpreter`: (Available to developer and LLM Agent in tool spec) Send a Python code to be executed in Azure Container Apps Dynamic Sessions and return the output in a JSON format.

`list_files`: (Available to developer and LLM Agent in tool spec) List the files available in a Session under the path `/mnt/data`.

`upload_file`: (Available to developer) Upload a file or a stream of data into a Session under the path `/mnt/data`.

`download_file`: (Available to developer) Download a file by its path relative to the path `/mnt/data` to the tool's hosting agent.
