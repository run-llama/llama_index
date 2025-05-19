# LlamaIndex Agent Integration: Azure

This package provides an Azure Foundry Agent integration for LlamaIndex. It allows you to leverage Azure's AI agent capabilities within your LlamaIndex applications.

## Installation

You can install the package via pip:

```bash
pip install llama-index-agent-azure
```

## Prerequisites

Before using this integration, ensure you have:

1.  An Azure account and a provisioned Azure OpenAI service or an Azure AI Project with an agent-compatible endpoint.
2.  The necessary environment variables set up for authentication. Typically, this involves:
    - `AZURE_PROJECT_ENDPOINT`: Your Azure AI Project endpoint.
    - Standard Azure authentication environment variables recognized by `DefaultAzureCredential` (e.g., `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`, or ensure you are logged in via Azure CLI).

## Usage

Here's a basic example of how to use the `AzureFoundryAgent`, with a toolset for function calling.

```python
from llama_index.agent.azure_agents import AzureFoundryAgent
from azure.ai.agents.models import FunctionTool, ToolSet
from dotenv import load_dotenv
import os


# Configure your Azure project endpoint
azure_project_endpoint = os.environ.get("AZURE_PROJECT_ENDPOINT")

if not azure_project_endpoint:
    raise ValueError("AZURE_PROJECT_ENDPOINT environment variable not set.")

# Define a sample tool (optional)
def get_weather(location: str) -> str:
    """Get the weather for a given location."""
    # This is a placeholder function. Replace with actual weather API call.
    return f"The weather in {location} is sunny."

toolset = ToolSet()
toolset.add(FunctionTool({get_weather}))

# Instantiate the agent
agent = AzureFoundryAgent(
    endpoint=azure_project_endpoint,
    model="gpt-4o",  # Specify your desired model
    name="my-azure-agent",
    instructions="You are a helpful assistant that can provide information and use tools.",
    verbose=True,
    toolset=toolset, # Optional: pass your defined toolset
    run_retrieve_sleep_time=2, # Time in seconds to wait between polling run status
)

# Chat with the agent
response = agent.chat("What is the capital of France and what is the weather there?")
print("Agent Response:", response.response)

# Important: Azure agents and threads are stateful resources on Azure.
# Remember to clean them up from the Azure portal or using the Azure SDK

```

### Key Parameters for `AzureFoundryAgent`

- `endpoint`: The endpoint URL for your Azure AI Project or compatible service.
- `model`: The identifier of the LLM model to be used by the agent (e.g., "gpt-4o", "gpt-35-turbo").
- `name`: A name for your agent instance.
- `instructions`: System instructions for the agent.
- `toolset` (Optional): An `azure.ai.agents.models.ToolSet` instance containing any tools you want the agent to use.
- `thread_id` (Optional): An existing thread ID to continue a conversation. If not provided, a new thread is created.
- `verbose` (Optional): Set to `True` for detailed logging.
- `run_retrieve_sleep_time` (Optional): The time in seconds to wait between polling the status of an agent run. Defaults to `1.0`.
