# LlamaIndex Tools for Commune (Email & SMS)

Connect your LlamaIndex agents to real email inboxes and SMS — powered by [Commune](https://commune.email).

## What is Commune?

Commune is email and SMS infrastructure built for AI agents. It gives agents a real inbox, the ability to send and receive email and SMS, and a simple API that wraps all the complexity of deliverability, threading, and parsing.

This package wraps the `commune-mail` Python SDK as a LlamaIndex `ToolSpec`, so your agents can:

- Read and search their email inbox
- Send emails on behalf of users or as autonomous actors
- Send SMS notifications or alerts
- Check API credit balance

## Installation

```bash
pip install llama-index-tools-commune
```

## Setup

Get an API key at [commune.email](https://commune.email) and set it as an environment variable:

```bash
export COMMUNE_API_KEY="your-api-key"
```

## Usage

### 1. Basic agent that reads and responds to emails

```python
import os
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.commune import CommuneToolSpec

commune_tools = CommuneToolSpec(api_key=os.environ["COMMUNE_API_KEY"])

agent = ReActAgent.from_tools(
    commune_tools.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
    verbose=True,
    system_prompt=(
        "You are an email assistant. Check the inbox for unread messages, "
        "understand what each sender needs, and reply helpfully."
    ),
)

response = agent.chat("Check for new emails and handle any that need a response.")
print(response)
```

### 2. Notification agent that sends updates

```python
import os
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.commune import CommuneToolSpec

commune_tools = CommuneToolSpec(api_key=os.environ["COMMUNE_API_KEY"])

agent = ReActAgent.from_tools(
    commune_tools.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
    verbose=True,
)

# Agent can send emails and SMS notifications
response = agent.chat(
    "Send an email to alice@example.com with subject 'Deployment Complete' "
    "telling her the production deploy finished successfully. Also send an SMS "
    "to +15551234567 confirming the same."
)
print(response)
```

### 3. Multi-agent communication via email

```python
import os
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.commune import CommuneToolSpec

# Two agents sharing an email-based communication channel
coordinator_tools = CommuneToolSpec(api_key=os.environ["COMMUNE_API_KEY"])

coordinator = ReActAgent.from_tools(
    coordinator_tools.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
    verbose=True,
    system_prompt=(
        "You are a coordinator agent. You receive task results from worker agents "
        "via email. Summarize their findings and email a final report to the team."
    ),
)

# Coordinator reads results from worker agents and compiles a report
response = coordinator.chat(
    "Search the inbox for emails from worker-agent@myapp.commune.email, "
    "read their reports, and email a summary to manager@example.com."
)
print(response)
```

## Available Tools

| Method | Description |
|---|---|
| `load_inbox` | Fetch recent emails from the inbox, optionally filtering to unread only |
| `search_emails` | Search emails by keyword, sender, or topic |
| `get_email` | Retrieve the full content of a specific email by ID |
| `send_email` | Compose and send an email to one or more recipients |
| `send_sms` | Send an SMS message to a phone number |
| `get_credits` | Check the current API credit balance |

## Environment Variables

| Variable | Description |
|---|---|
| `COMMUNE_API_KEY` | Required. Your Commune API key. |

## Links

- [Commune Docs](https://commune.email/docs)
- [commune-mail on PyPI](https://pypi.org/project/commune-mail/)
- [LlamaIndex Docs](https://docs.llamaindex.ai)
