# LlamaIndex Readers: x402 CleanWeb Data Reader

Official LlamaIndex integration for **CleanWeb Studio** (`x402-cleanweb-agent`).

This loader ingests web pages at scale, stripping boilerplate HTML, navigation scripts, and advertisements to achieve an **87% reduction in token consumption** while protecting RAG pipelines against **indirect prompt injection**.

---

## 🌟 Key Features

- **87% Token Compression**: Reduces 25,000-token HTML pages to ~3,000 clean Markdown tokens.
- **Indirect Prompt Injection Defense**: Strips hidden CSS containers (`display:none`, `font-size:0`), zero-width steganography, and malicious tracking pixels.
- **Multimodal Alt-Text Preservation**: Keeps descriptive image captions and charts (`![alt](url)`) for multimodal models while dropping 1x1 tracking beacons.
- **Zero-Overhead SPA Hydration**: Ingests pre-rendered state from Next.js (`__NEXT_DATA__`), Nuxt.js (`__NUXT_DATA__`), and Schema.org JSON-LD in sub-milliseconds without heavy headless browsers.
- **Verifiable Provenance**: Attaches canonical URL, title, reading time, and SHA-256 content fingerprints directly into LlamaIndex `Document.metadata`.
- **Autonomous Web3 x402 Settlement**: Native machine-to-machine USDC micropayments on Base, Polygon, Arbitrum, and Solana with a zero-auth sandbox trial.

---

## 🚀 Installation

```bash
pip install llama-index-readers-x402-cleanweb
```

---

## 💻 Quickstart Usage

```python
from llama_index.readers.x402_cleanweb import X402CleanWebReader
from llama_index.core import VectorStoreIndex

# 1. Initialize reader (optionally pass pre-funded Vault Session Key)
reader = X402CleanWebReader(vault_key="your_vault_session_key")

# 2. Ingest clean web documents
documents = reader.load_data(
    urls=[
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://news.ycombinator.com/"
    ],
    preserve_alt_text=True,
    dynamic_render=False
)

# 3. Build Vector Index directly
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What are the key discussions?")
print(response)
```

---

## 📄 License
MIT License.

