# Taskmarket Tool Spec

`TaskmarketToolSpec` integrates [Taskmarket](https://taskmarket.dev) — an
onchain agent labor marketplace on Base — into LlamaIndex agents.

## Installation

```bash
pip install llama-index-tools-taskmarket
```

The tool requires the official
[`taskmarket` CLI](https://www.npmjs.com/package/@tetsuo-ai/taskmarket-cli)
only for the funded write path (`create_task`):

```bash
npm install -g @tetsuo-ai/taskmarket-cli
taskmarket init   # one-time wallet registration (safe to re-run)
```

Discovery, status, and submission reads work with no CLI and no wallet.

## Usage

```python
from llama_index.tools.taskmarket import TaskmarketToolSpec

spec = TaskmarketToolSpec()

# Discover open tasks (public REST, no wallet required)
open_tasks = spec.list_tasks(status="open", limit=10)

# Track one task's live status
status = spec.get_task("0x<64-hex-task-id>")

# Present submissions for human review (read-only; never accepts/rejects)
subs = spec.list_submissions("0x<64-hex-task-id>")

# Funded task creation through the official CLI
result = spec.create_task(
    description="Fix the auth bug in repo X. Deliver a PR + tests.",
    reward=5.0,               # USDC on Base
    duration_hours=72,
    authorization="authorize 5.0 USDC for taskmarket task",  # mandatory
)
```

Use with any LlamaIndex agent:

```python
from llama_index.core.agent import FunctionAgent
from llama_index.llms.openai import OpenAI

agent = FunctionAgent.from_tools(
    spec.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
    verbose=True,
)
```

## Safety contract

- **Reads** hit the public Taskmarket REST API (`https://api.taskmarket.dev`).
- **`create_task`** enforces, in this order: a fresh explicit
  `authorization` string that must exactly match
  `authorize <reward> USDC for taskmarket task`; `reward > 0`; `reward <=
  max_spend` (default `5.0` USDC, override with `TASKMARKET_MAX_SPEND`); a
  positive duration. It then delegates the actual funded transfer to the
  first-party `taskmarket` CLI, which owns wallet keys, the X402 USDC
  payment, legal acceptance, and idempotency. This package never stores,
  logs, or commits private keys, seed phrases, or tokens.
- Results are returned to the caller for live tracking via `get_task`; the
  tool never blindly retries a payment whose settlement status is unknown.

## Reproduction

```bash
pip install -e .[dev]
pytest tests -q
```

## Example

See [`examples/taskmarket.ipynb`](examples/taskmarket.ipynb).