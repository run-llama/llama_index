# LlamaIndex Reader: DataSinking

Load full-text financial reports (balance sheet, income statement, cash flow + notes) as LlamaIndex `Document`s for your RAG pipeline. Makes DataSinking the first step of "build a financial-report RAG".

## Installation

```bash
pip install llama-index-readers-datasinking
```

## Usage (20-line Colab)

```python
!pip install llama-index-readers-datasinking

from llama_index.readers.datasinking import DataSinkingReader

# Get a free key at datasink.ing; leave empty for the public quota (31 docs / 7 days / IP)
reader = DataSinkingReader(api_key="your key")
docs = reader.load_data("AAPL", limit=3)   # Apple's 3 latest filings

from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(docs)
print(index.as_query_engine().query("What was Apple's net income last year?"))
```

## Supported symbols

- US `AAPL` · China `600519.SS` · Japan `7203.T` · Korea `005930.KS` · Taiwan `2330.TW` · UK `VOD.L`
- Don't know the ticker? Search by name: `GET https://api.datasink.ing/search?q=Apple`
- Each report is a `Document` with metadata `symbol` / `report_period` / `doc_type`

API docs: https://datasink.ing/docs
