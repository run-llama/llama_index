# LlamaIndex Tools Integration: Repull

[Repull](https://repull.dev) is a unified API in front of 50+ vacation-rental
property-management systems and the Airbnb / Booking.com / VRBO / Plumguide
channels. This tool spec wraps the typed
[`repull-sdk`](https://pypi.org/project/repull-sdk/) Python client so a
LlamaIndex agent can answer questions about a property manager's portfolio,
look up reservations, explore markets for pricing data, and kick off OAuth
onboarding for a new channel account.

The exposed surface is read-only customer-facing endpoints plus the Connect
session-creation entry point. Admin / billing / superadmin endpoints are
intentionally not exposed here.

## Install

```bash
pip install llama-index llama-index-tools-repull
```

## Get an API key

Sign up at [repull.dev](https://repull.dev) and grab an API key from the
dashboard. Sandbox keys start with `sk_test_`, live keys with `sk_live_`.

## Usage

```python
import os

from llama_index.tools.repull import RepullToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

repull = RepullToolSpec(api_key=os.environ["REPULL_API_KEY"])

agent = FunctionAgent(
    tools=repull.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
    system_prompt=(
        "You are an assistant for a vacation-rental property manager. "
        "Use the Repull tools to answer questions about their properties, "
        "reservations, conversations, and pricing markets."
    ),
)

print(await agent.run("How many active properties do I have, and which markets am I in?"))
print(await agent.run("Show me Airbnb reservations checking in next week."))
print(await agent.run("What's the ADR distribution in Lisbon right now?"))
```

## Tools

| Tool                       | What it does                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `list_properties`          | Cursor-paginated list of properties in the workspace, filterable by status.           |
| `get_property`             | Fetch one property by Repull id.                                                      |
| `list_reservations`        | List reservations with filters (platform, status, listing_id, check-in date range).   |
| `list_markets`             | Markets the customer operates in, with per-city KPIs (ADR vs market, occupancy, etc). |
| `search_markets`           | Paginated discovery catalog — search every market Repull tracks globally.             |
| `get_market`               | Deep-dive one market: price distribution, comps, demand, monthly benchmarks.          |
| `list_conversations`       | Guest message threads, filterable by channel and status.                              |
| `create_connect_session`   | Mint a white-label OAuth Connect URL for linking a new channel account.               |

Each method's docstring is the LLM-facing description — see
[`base.py`](llama_index/tools/repull/base.py) for the full prompts.

## Pagination

Every list endpoint returns a dict shaped like:

```python
{
    "data": [...],
    "pagination": {
        "has_more": True,
        "next_cursor": "eyJ...",
        "total": 412,
    },
}
```

Loop until `pagination.has_more` is `False`, threading `pagination.next_cursor`
back as the next call's `cursor` arg. Cursors are opaque base64 — never parse
or construct them by hand.

## References

- [`repull-sdk`](https://pypi.org/project/repull-sdk/) — the underlying typed
  Python client.
- [api.repull.dev](https://api.repull.dev) — API base URL and OpenAPI spec.
- [repull.dev/docs](https://repull.dev/docs) — product docs.

## License

MIT.
