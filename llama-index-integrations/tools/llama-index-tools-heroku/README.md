# LlamaIndex Tools Integration: Heroku

This package provides LlamaIndex tools that integrate with the Heroku Agents API, enabling LlamaIndex agents to use Heroku's built-in tools for database queries, code execution, and more.

## Installation

```bash
pip install llama-index-tools-heroku
```

## Usage

### Basic Setup

```python
from llama_index.tools.heroku import HerokuToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.heroku import Heroku

# Initialize the tool spec
tool_spec = HerokuToolSpec(
    api_key="your-heroku-inference-key",
    app_name="your-heroku-app-name",
)

# Get tools for use with an agent
tools = tool_spec.to_tool_list()

# Create an agent
llm = Heroku(model="claude-4-5-sonnet", api_key="your-key")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

# Use the agent
response = agent.chat("Run a SQL query to count the users in the database")
print(response)
```

### Available Tools

The package provides the following tools:

- **run_sql**: Execute SQL queries on Heroku Postgres
- **run_python**: Execute Python code in a sandboxed environment
- **run_javascript**: Execute JavaScript/Node.js code
- **get_app_info**: Get information about a Heroku app

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | Required | Your Heroku Inference API key |
| `app_name` | str | Required | The Heroku app name to interact with |
| `base_url` | str | `"https://us.inference.heroku.com"` | Heroku Inference API base URL |
| `timeout` | float | `120.0` | Request timeout in seconds |

## Example: Database Query Agent

```python
from llama_index.tools.heroku import HerokuToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.heroku import Heroku
import os

# Setup
api_key = os.getenv("INFERENCE_KEY")
tool_spec = HerokuToolSpec(api_key=api_key, app_name="my-app")
llm = Heroku(model="claude-4-5-sonnet", api_key=api_key)

# Create agent with database tools
agent = ReActAgent.from_tools(
    tool_spec.to_tool_list(),
    llm=llm,
    verbose=True,
)

# Query the database naturally
response = agent.chat(
    "What are the top 10 products by revenue? "
    "Show me the product name and total revenue."
)
print(response)
```

## Example: Code Execution Agent

```python
from llama_index.tools.heroku import HerokuToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.heroku import Heroku

tool_spec = HerokuToolSpec(api_key=api_key, app_name="my-app")
agent = ReActAgent.from_tools(tool_spec.to_tool_list(), llm=llm)

# Execute code through natural language
response = agent.chat(
    "Calculate the Fibonacci sequence up to 100 and return the results"
)
print(response)
```

## Security Considerations

- SQL queries are executed with appropriate permissions based on your Heroku app's database configuration
- Code execution happens in a sandboxed environment
- API keys should be stored securely using environment variables
- Consider using read-only database credentials for query-only use cases

## License

MIT
