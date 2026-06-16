# LlamaIndex Tools: BuyWhere

This tool connects a LlamaIndex agent to the [BuyWhere](https://buywhere.ai) product catalog
API. BuyWhere indexes 1.5M+ products across Shopee, Lazada, Amazon, Walmart, and 20+
retailers in Southeast Asia and the US, with live price comparison and affiliate deep-links.

## Why BuyWhere

- **Cross-border, cross-merchant search.** One query, one ranked result set across every
  supported retailer — no per-merchant integration work.
- **Honest price comparison.** Side-by-side offers with shipping, currency, and stock state.
- **Affiliate-ready.** Generate tracked deep-links for any merchant in one call.

## Installation

```bash
pip install llama-index-tools-buywhere
```

## Usage

You will need a BuyWhere API key. Sign up at <https://buywhere.ai/developers> to get one
(free tier includes 1,000 requests/day).

```python
from llama_index.tools.buywhere import BuyWhereToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tool_spec = BuyWhereToolSpec(
    api_key="your-buywhere-api-key",
    country="US",
)

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
    system_prompt=(
        "You are a shopping concierge. Use the buywhere tools to search the catalog, "
        "compare prices, and return a shortlist with the best deal highlighted."
    ),
)

print(await agent.run("What's the best gaming laptop under $1500 right now?"))
print(await agent.run("Compare iPhone 17 Pro prices across Amazon and Shopee."))
```

## Available tools

`search_products(query, limit=5)` — Search the catalog. Returns ranked results with title, price, merchant, and URL.

`get_product(product_id)` — Fetch full product details by id.

`compare_prices(product_id)` — Return every live offer for a product across all supported merchants.

`get_affiliate_link(product_id, merchant=None)` — Generate a tracked deep-link (defaults to the lowest-priced merchant).

`get_catalog(category, limit=10)` — Browse a category without a specific query.

## Configuration

| Field         | Default | Description                                                                 |
|---------------|---------|-----------------------------------------------------------------------------|
| `api_key`     | —       | BuyWhere API key (required).                                                |
| `marketplace` | `None`  | Restrict searches to one marketplace (e.g. `amazon`, `shopee`). `None` = all.|
| `country`     | `US`    | Country for pricing/currency (ISO 3166-1 alpha-2).                          |

## Links

- BuyWhere: <https://buywhere.ai>
- API docs: <https://api.buywhere.ai/docs>
- MCP server: <https://github.com/BuyWhere/buywhere-mcp>
- npm SDK: <https://www.npmjs.com/package/@buywhere/sdk>
