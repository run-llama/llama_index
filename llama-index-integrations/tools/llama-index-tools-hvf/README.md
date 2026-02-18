# HVF Tool (Hudson Valley Forestry API)

```bash
pip install llama-index-tools-hvf
```

This tool allows an LLM agent to interact with the
[Hudson Valley Forestry](https://www.hudsonvalleyforestry.com) public API.
It supports health checks, retrieving available forestry services, and
submitting service inquiries for both residential and commercial clients.

## Available Tools

| Tool | Description |
|------|-------------|
| `health_check` | Verify the HVF API is reachable and healthy |
| `get_services` | List available forestry services (optionally filtered by category) |
| `submit_residential_inquiry` | Submit a residential service inquiry / contact form |
| `submit_commercial_inquiry` | Submit a commercial project inquiry / contact form |

## Usage

```python
from llama_index.tools.hvf import HVFToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

hvf_tool = HVFToolSpec()

agent = FunctionAgent(
    tools=hvf_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

# Check API health
print(await agent.run("Is the Hudson Valley Forestry API currently available?"))

# List services
print(await agent.run("What forestry services does Hudson Valley Forestry offer?"))

# Submit a residential inquiry
print(await agent.run(
    "Submit a tree removal inquiry from Jane Doe (jane@example.com, 845-555-0001) "
    "at 123 Forest Rd, Woodstock NY. She needs 3 large oaks removed from her 2.5 acre lot."
))

# Submit a commercial inquiry
print(await agent.run(
    "Submit a commercial land clearing inquiry for Acme Land Co. Contact: Bob Smith "
    "(bob@acme.com, 845-555-0099), 500 Industrial Blvd Kingston NY. "
    "Need 40 acres cleared by Spring 2026."
))
```

## API Reference

**Base URL**: `https://app.hudsonvalleyforestry.com/api`

### Endpoints used

| Method | Path | Tool |
|--------|------|------|
| GET | `/health` | `health_check` |
| GET | `/services` | `get_services` |
| POST | `/inquiry` | `submit_residential_inquiry` |
| POST | `/inquiry/commercial` | `submit_commercial_inquiry` |

## Configuration

```python
# Custom base URL or timeout
hvf_tool = HVFToolSpec(
    base_url="https://app.hudsonvalleyforestry.com/api",
    timeout=60,
)
```

This loader is designed to be used as a way to interact with the
Hudson Valley Forestry API in an LLM Agent context.
