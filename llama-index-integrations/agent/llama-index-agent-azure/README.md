# LlamaIndex Azure Foundry Agent Integration

This package provides an Azure Foundry Agent integration for LlamaIndex. It allows you to leverage [Azure AI Agent Service](https://learn.microsoft.com/en-us/azure/ai-services/agents/overview) capabilities within your LlamaIndex applications. The provided `AzureFoundryAgent` inherits `BaseWorkflowAgent` from LlamaIndex, making it compatible with workflow-based multi-agent orchestration.

> **About Azure AI Agent Service**
>
> [Azure AI Agent Service](https://learn.microsoft.com/en-us/azure/ai-services/agents/overview) is a fully managed service designed to empower developers to securely build, deploy, and scale high-quality, and extensible AI agents without needing to manage the underlying compute and storage resources.

## Installation

You can install the package via pip:

```bash
pip install llama-index-agent-azure
```

or if working from source:

```bash
cd llama_index/llama-index-integrations/agent/llama-index-agent-azure
pip install -e .
```

You may also want to install `python-dotenv` if you plan to use a `.env` file for environment variables:

```bash
pip install python-dotenv
```

## Prerequisites

Before using this integration, ensure you have:

1.  An Azure account and a provisioned Azure OpenAI service or an Azure AI Project with an agent-compatible endpoint.
2.  The necessary environment variables set up for authentication. Typically, this involves:
    - `AZURE_PROJECT_ENDPOINT`: Your Azure AI Project endpoint.
    - Standard Azure authentication environment variables recognized by `DefaultAzureCredential` (e.g., `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`, or ensure you are logged in via Azure CLI).

## Usage

Here's a basic example of how to use the `AzureFoundryAgent` with a function tool for function calling.

```python
from llama_index.agent.azure_foundry_agent import AzureFoundryAgent
from dotenv import load_dotenv
import os

load_dotenv()

# Configure your Azure project endpoint
azure_project_endpoint = os.environ.get("AZURE_PROJECT_ENDPOINT")

if not azure_project_endpoint:
    raise ValueError("AZURE_PROJECT_ENDPOINT environment variable not set.")


# Define a sample tool (optional)
def get_weather(location: str) -> str:
    """Get the weather for a given location."""
    # This is a placeholder function. Replace with actual weather API call.
    return f"The weather in {location} is sunny."


# Instantiate the agent
# Note: The `model` parameter refers to a model deployment that should already be created in your Azure AI Project.
agent = AzureFoundryAgent(
    endpoint=azure_project_endpoint,
    model="gpt-4o",  # Specify your deployed model name
    name="my-azure-agent",
    instructions="You are a helpful assistant that can provide information and use tools.",
    verbose=True,
    tools=[get_weather],  # Pass your defined tools as a list
    run_retrieve_sleep_time=2,  # Time in seconds to wait between polling run status
)

# Run the agent
response = await agent.run(
    "What is the capital of France and what is the weather there?"
)
print("Agent Response:", response)
```

Example of using multimodal input with the agent:

```python
# Example: Multimodal input (text + image)
# This works with multimodal-capable models such as gpt-4o
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock

multimodal_msg = ChatMessage(
    role="user",
    blocks=[
        TextBlock(text="Describe what you see in this image."),
        ImageBlock(url="https://example.com/sample-image.png"),
    ],
)
multimodal_response = await agent.run(multimodal_msg)
print("Multimodal Agent Response:", multimodal_response)


# Important: Azure agents and threads are stateful resources on Azure.
# Remember to clean them up from the Azure portal or using the Azure SDK
# await agent._client.agents.delete_agent(agent_id=agent._agent.id)
# await agent._client.agents.threads.delete(thread_id=agent._thread_id)
```

### Key Parameters for `AzureFoundryAgent`

- `endpoint`: The endpoint URL for your Azure AI Project or compatible service.
- `model`: The identifier of the LLM model to be used by the agent (e.g., "gpt-4o", "gpt-35-turbo").
- `name`: A name for your agent instance.
- `instructions`: System instructions for the agent.
- `tools` (Optional): A list of Python functions to be used as tools by the agent.
- `thread_id` (Optional): An existing thread ID to continue a conversation. If not provided, a new thread is created.
- `verbose` (Optional): Set to `True` for detailed logging.
- `run_retrieve_sleep_time` (Optional): The time in seconds to wait between polling the status of an agent run. Defaults to `1.0`.

## Troubleshooting

- **Missing Environment Variables**: Ensure `AZURE_PROJECT_ENDPOINT` and Azure credentials are set in your environment or `.env` file.
- **Resource Cleanup**: Always delete agents and threads after use to avoid resource leaks and unnecessary Azure charges.
- **Dependency Issues**: Make sure all required packages are installed, including `python-dotenv` if using `.env` files.

For more details, see the [Azure AI Agent Service documentation](https://learn.microsoft.com/en-us/azure/ai-services/agents/overview).
