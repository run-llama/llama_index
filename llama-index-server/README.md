# LlamaIndex Server

LlamaIndexServer is a FastAPI application that allows you to quickly launch your workflow as an API server.

## Installation

```bash
pip install llama-index-server
```

## Usage

```python
# main.py
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Workflow
from llama_index.core.tools import FunctionTool
from llama_index.server import LlamaIndexServer


# Define a factory function that returns a Workflow or AgentWorkflow
def create_workflow() -> Workflow:
    def fetch_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    return AgentWorkflow.from_tools(
        tools=[
            FunctionTool.from_defaults(
                fn=fetch_weather,
            )
        ]
    )


# Create an API server the workflow
app = LlamaIndexServer(
    workflow_factory=create_workflow  # Supports Workflow or AgentWorkflow
)
```

## Running the server

- In the same directory as `main.py`, run the following command to start the server:

  ```bash
  fastapi dev
  ```

- Making a request to the server

  ```bash
  curl -X POST "http://localhost:8000/api/chat" -H "Content-Type: application/json" -d '{"message": "What is the weather in Tokyo?"}'
  ```

- See the API documentation at `http://localhost:8000/docs`
