# LlamaIndex Tools Integration: UniRate

`llama-index-tools-unirate` gives a LlamaIndex agent live **currency exchange rates**,
**currency conversion**, the **list of supported currencies**, and **VAT rates** via the
[UniRate API](https://unirateapi.com). A free API key is all you need.

## Installation

```bash
pip install llama-index-tools-unirate
```

## Usage

```python
from llama_index.tools.unirate import UnirateToolSpec
from llama_index.agent.openai import OpenAIAgent

tool_spec = UnirateToolSpec(api_key="your-unirate-api-key")

agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("What is 250 USD in EUR right now?")
agent.chat("What is the current exchange rate from GBP to JPY?")
agent.chat("What is the VAT rate in Germany?")
```

The tool spec exposes four functions:

| Function | Description |
|---|---|
| `get_exchange_rate(from_currency, to_currency)` | Current rate between two currencies |
| `convert_currency(amount, from_currency, to_currency)` | Convert an amount at the current rate |
| `list_supported_currencies()` | All supported currency codes |
| `get_vat_rates(country=None)` | VAT rates for one or all countries |

This tool is a wrapper around the [`unirate-api`](https://pypi.org/project/unirate-api/)
Python client. This loader is designed to be used as a way to load data as a Tool in a
Agent. See [here](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools/llama-index-tools-unirate) for examples.
