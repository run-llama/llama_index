# Verifly Tool

[Verifly](https://verifly.email) is an agent-native **email verification** API.
It tells you whether an email address is real and safe to send to — checking
deliverability (syntax, domain, MX, SMTP) and risk flags (disposable, role,
catch-all, free provider) — and returns a verdict with a send / do-not-send
recommendation.

This tool lets a LlamaIndex agent verify an address before adding it to a list,
confirming a signup, or firing off a message.

## Installation

```bash
pip install llama-index-tools-verifly
```

## Getting an API key

Create an account and grab a key at [verifly.email](https://verifly.email).
Verifly is built for autonomous workflows, so an agent can also self-onboard for
a key (with free starter credits) and no human in the loop via the autonomous
registration endpoint. Set it as an environment variable:

```bash
export VERIFLY_API_KEY="vf_..."
```

## Usage

```python
from llama_index.tools.verifly import VeriflyToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

verifly_tool = VeriflyToolSpec(api_key="vf_...")  # or rely on VERIFLY_API_KEY

agent = FunctionAgent(
    tools=verifly_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
)

await agent.run("Is lead@example.com a deliverable email address?")
```

Call it directly:

```python
from llama_index.tools.verifly import VeriflyToolSpec

verifly_tool = VeriflyToolSpec()  # reads VERIFLY_API_KEY from the environment
doc = verifly_tool.verify_email("lead@example.com")

print(doc.text)            # e.g. "lead@example.com: deliverable - recommendation: send"
print(doc.metadata)        # full structured result: is_valid, result, reason, details, ...
```

## Available Functions

`verify_email`: Verify a single email address. Returns a `Document` whose `text`
is a short human-readable verdict and whose `metadata` holds the full
structured Verifly result (`is_valid`, `result`, `reason`, `details`,
`recommendation`, `credits`).

This loader is designed to be used as a Tool in an Agent.
