# Finance Chat Llama Pack based on OpenAIAgent

This LlamaPack implements a hierarchical agent based on LLM for financial chat and information extraction purposed.

LLM agent is connected to various open financial apis as well as daily updated SP500 postgres SQL database storing
opening & closing price, volume as well as past earnings.

Based on the query, the agent reasons and routes to available tools or runs SQL query to retrieve information and
combine information to answer.

### Installation

```bash
pip install llama-index llama-index-packs-finchat
```

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack FinanceChatPack --download-dir ./finchat_pack
```

You can then inspect the files at `./finchat_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./finchat_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

FinanceChatPack = download_llama_pack("FinanceChatPack", "./finchat_pack")
```

To use this tool, you'll need a few API keys:

- POLYGON_API_KEY -- <https://polygon.io/>
- FINNHUB_API_KEY -- <https://finnhub.io/>
- ALPHA_VANTAGE_API_KEY -- <https://www.alphavantage.co/>
- NEWSAPI_API_KEY -- <https://newsapi.org/>
- POSTGRES_DB_URI -- 'postgresql://postgres.xhlcobfkbhtwmckmszqp:fingptpassword#123@aws-0-us-east-1.pooler.supabase.com:5432/postgres' (You can also host your own postgres SQL DB with the same table signatures. To use different signatures, modification is required in giving query examples for SQL code generation.)

```python
fin_agent = FinanceChatPack(
    POLYGON_API_KEY,
    FINNHUB_API_KEY,
    ALPHA_VANTAGE_API_KEY,
    NEWSAPI_API_KEY,
    OPENAI_API_KEY,
)
```

From here, you can use the pack, or inspect and modify the pack in `./finchat_pack`.

The `run()` function chats with the agent and sends the response of the input query.

```python
response = fin_agent.run("<query>")
```
