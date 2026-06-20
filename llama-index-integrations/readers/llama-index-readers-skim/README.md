# LlamaIndex Readers Integration: Skim

[Skim](https://skim402.com) is an x402-native clean reader API for AI agents.
`SkimReader` fetches any URL and returns a LlamaIndex `Document` of clean,
agent-ready Markdown plus structured metadata (title, byline, published date,
language, excerpt), with nav, ads, and boilerplate stripped.

Skim has no API keys and no signup. Each read is paid automatically over the
[x402 protocol](https://x402.org): $0.002 per call in USDC on Base, signed
locally by a wallet you control. The private key never leaves your machine - it
only signs an EIP-3009 USDC authorization.

## Installation

```bash
pip install llama-index-readers-skim
```

This pulls in the x402 client with EVM support (`x402[evm]`), `eth-account`, and
`requests`.

## Setup

Set `SKIM_WALLET_PRIVATE_KEY` to the hex private key of a Base wallet funded with
a little USDC. Use a dedicated wallet, never your personal one.

```bash
export SKIM_WALLET_PRIVATE_KEY=0x...
```

## Usage

```python
from llama_index.readers.skim import SkimReader

reader = SkimReader()  # reads SKIM_WALLET_PRIVATE_KEY from the env
documents = reader.load_data(urls=["https://en.wikipedia.org/wiki/HTTP_402"])

print(documents[0].text)      # clean Markdown
print(documents[0].metadata)  # title, byline, published date, language, ...
```

`load_data` accepts a single URL string or a list of URLs and returns one
`Document` per URL.

## Configuration

`SkimReader` accepts these optional parameters:

- `private_key` (str): Base wallet hex private key. Falls back to the
  `SKIM_WALLET_PRIVATE_KEY` environment variable.
- `base_url` (str): Skim API base URL. Defaults to `https://skim402.com`.
- `max_price_usd` (float): Hard per-call price cap in USD. The wallet refuses to
  sign for anything above this. Defaults to `0.01` (Skim is `$0.002`).
- `include_metadata` (bool): When `True` (default), populate each `Document`'s
  metadata with the page metadata returned by Skim.
- `timeout` (float): Per-request timeout in seconds. Defaults to `60`.
