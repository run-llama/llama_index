---
title: Offline Agent Testing With AiEnsured agentest
---

[AiEnsured agentest](https://github.com/srpadmanabhuni/agentest) (`agentest-ai`) is an Agentic AI Testing platform that evaluates recorded agent traces offline using **165 deterministic checks** over OpenTelemetry GenAI traces, A2A AgentCards, and MCP tool annotations — without an LLM judge or API key.

## Installation

```sh
pip install agentest-ai
```

## Quickstart

Record or import agent runs offline and check whether the agent actually completed what it claimed:

```python
import agentest

with agentest.record(
    "llamaindex_refund_agent",
    "Refund order ORD-1042, item arrived damaged",
) as rec:
    rec.tool(
        "lookup_order",
        {"order_id": "ORD-1042"},
        {"status": "delivered", "total": 49.0},
        reason="check order status before refunding",
    )
    rec.tool(
        "issue_refund",
        {"order_id": "ORD-1042", "amount": 49.0},
        {"error": "gateway timeout"},
        error=True,
        reason="item arrived damaged",
    )
    rec.answer("Done. I refunded $49.00 for order ORD-1042.")

result = rec.check()
print(result)
# otlp_agent: 16 pass, 5 fail (T1-002, T1-005, T1-018, X-008, X-012)
```

Use `agentest.assert_passes(result)` inside `pytest` to fail CI whenever an applicable check fails.

## Exporting LlamaIndex Workflow Runs

Capture `ToolCallResult` events and `AgentOutput` from `AgentWorkflow.run(...)` without importing `agentest` at runtime:

```python
from llama_index.core.agent.workflow import ToolCallResult

trace = []
handler = workflow.run(user_msg=user_msg)
async for event in handler.stream_events():
    if isinstance(event, ToolCallResult):
        trace.append(
            {
                "kind": "tool",
                "name": event.tool_name,
                "data": {
                    "args": event.tool_kwargs,
                    "result": event.tool_output.raw_output,
                    "error": event.tool_output.is_error,
                },
            }
        )
output = await handler
```

Then convert to OpenTelemetry JSONL and evaluate in CI:

```sh
agentest std import workflow_runs.jsonl --map map.json --tools tools.json --service my_workflow --out trace.otlp.jsonl
agentest std run --otlp trace.otlp.jsonl --card agent-card.json --tools tools.json --strict
```

## Links

- [Agentest Evaluation Notebook](/python/examples/evaluation/agentest)
- [agentest-ai on PyPI](https://pypi.org/project/agentest-ai/)
- [AiEnsured agentest on GitHub](https://github.com/srpadmanabhuni/agentest)
- [LlamaIndex Integration Guide in agentest](https://github.com/srpadmanabhuni/agentest/blob/main/docs/integrations/llamaindex.md)
