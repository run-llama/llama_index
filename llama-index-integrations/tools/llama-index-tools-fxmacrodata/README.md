# FXMacroData Tool

Official-source macroeconomic, FX and central-bank data for 18 currencies, from
[FXMacroData](https://fxmacrodata.com).

The value for an agent is aggregation. Answering "what did US core inflation print at, and
when is the next release" otherwise means knowing which of eighteen official publishers to
call and how each one formats its data. `get_latest_macro_snapshot` returns the newest print
of every indicator for an economy in a single request, and every observation carries the
instant it was published, so a value is never presented as though it were known before its
release.

**No API key is required for USD.** A key widens the history window (anonymous access returns
the most recent 90 days) and unlocks the other seventeen currencies plus FX rates, rate
differentials, COT positioning and commodities. Set `FXMACRODATA_API_KEY` or pass `api_key`.

## Installation

```bash
pip install llama-index-tools-fxmacrodata
```

## Usage

```python
from llama_index.tools.fxmacrodata import FXMacroDataToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = FXMacroDataToolSpec()

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(
    await agent.run(
        "What is the latest US inflation print, when was it published, "
        "and what is due next?"
    )
)
```

## Available tools

| Tool | Purpose |
| --- | --- |
| `search_indicators` | Every indicator slug published for a currency, with units and coverage |
| `get_latest_macro_snapshot` | The newest print of every indicator, in one request |
| `get_indicator_history` | One indicator's published history |
| `get_release_calendar` | Upcoming scheduled releases with publication times |
| `get_central_bank_headlines` | Official central-bank press releases |
| `get_fx_rate` | Official reference exchange rates for a pair |
| `get_rate_differential` | Policy rate differential, the first look at carry |
| `get_cot_positioning` | CFTC Commitment of Traders positioning |
| `get_commodity_prices` | Latest tracked commodity prices |
| `get_market_sessions` | Which FX sessions are open now |
| `get_risk_sentiment` | Cross-asset risk sentiment reading |

Call `search_indicators` first when the indicator slug is not already known; it is the
discovery step that makes the rest usable.

## Notes

- The API key is sent as an `X-API-Key` header rather than a query parameter, so it does not
  land in proxy or server access logs.
- A `401` or `403` returns a message explaining that a key is required, rather than surfacing
  as an outage, so the agent can fall back to USD instead of retrying blindly.
- `limit` is capped at 100 by the API.

This loader is designed to be used as a way to load data as a Tool in a Agent.
See [here](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools) for examples.
