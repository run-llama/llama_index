# llama-index-readers-azure-tax-forms

[![CI](https://github.com/zavera/llama-index-readers-azure-tax-forms/actions/workflows/ci.yml/badge.svg)](https://github.com/zavera/llama-index-readers-azure-tax-forms/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

A [LlamaIndex](https://github.com/run-llama/llama_index) reader that extracts structured key-value pairs from IRS tax form PDFs using **Azure Document Intelligence**.

Built and production-tested at [Callisto Tech](https://github.com/zavera) as part of a financial aid advising platform processing real tax documents at scale.

---

## Supported Forms

| Form | Description |
|------|-------------|
| Form 1040 | Individual income tax return |
| W-2 | Wage and tax statement |
| Schedule C | Profit or loss from business |
| Schedule E | Supplemental income and loss |
| Schedule K-1 | Partner's / shareholder's share of income |
| Form 1065 | U.S. return of partnership income |
| Form 1120 / 1120-S | Corporate income tax return |

---

## Installation

```bash
pip install llama-index-readers-azure-tax-forms
```

---

## Quick Start

```python
from llama_index_readers_azure_tax_forms import AzureTaxFormReader

reader = AzureTaxFormReader(
    endpoint="https://my-resource.cognitiveservices.azure.com/",
    api_key="YOUR_AZURE_DI_KEY",
    max_concurrent=12,
)

# Single file
docs = reader.load_data("path/to/1040.pdf")

# Multiple files — processed concurrently, gate limits Azure DI calls
docs = reader.load_data(["1040.pdf", "w2.pdf", "schedule_c.pdf"])

# From raw bytes (S3, blob storage, database, etc.)
docs = reader.load_data_from_bytes([
    ("1040.pdf", open("1040.pdf", "rb").read()),
    ("w2.pdf",   open("w2.pdf",   "rb").read()),
])
```

---

## Real Extraction Example

The following output was produced by running this reader against official IRS Form 1040 and W-2 templates filled with **fictional test data**.

### Input — Form 1040 (filled with fake data)

```python
docs = reader.load_data("samples/f1040_filled.pdf")
doc = docs[0]
print(doc.text)
print(doc.metadata)
```

### Output — `doc.text` (key | value per line)

```
Your first name and middle initial | James
Last name | Harrington
Your social security number | XXX-XX-1234
Home address | 742 Evergreen Terrace
City, town, or post office | Springfield
State | IL
ZIP code | 62701
Wages, salaries, tips, etc. | 82000
Ordinary dividends | 1200
Total income | 83200
Adjusted gross income | 83200
Standard deduction | 13850
Taxable income | 69350
Tax | 11500
Total tax | 11500
Federal income tax withheld from Form(s) W-2 | 13200
Total payments | 13200
Amount of line 33 you want refunded to you | 1700
```

### Output — `doc.metadata`

```json
{
  "document_id": "samples/f1040_filled.pdf",
  "form_type": "1040",
  "kv_count": 19,
  "stage": "STAGE-0",
  "di_calls": 1,
  "az_di_ms": 1843,
  "total_ms": 1921,
  "error": null
}
```

### Input — W-2 (filled with fake data)

```
Employee's social security number | XXX-XX-1234
Employer identification number (EIN) | 12-3456789
Employer's name, address, and ZIP code | Acme Corporation
Employee's first name and initial | James
Employee's last name | Harrington
Wages, tips, other compensation | 82000
Federal income tax withheld | 13200
Social security wages | 82000
Social security tax withheld | 5084
Medicare wages and tips | 82000
Medicare tax withheld | 1189
```

---

## Use in a LlamaIndex RAG Pipeline

```python
from llama_index_readers_azure_tax_forms import AzureTaxFormReader
from llama_index.core import VectorStoreIndex

reader = AzureTaxFormReader(
    endpoint="https://my-resource.cognitiveservices.azure.com/",
    api_key="YOUR_AZURE_DI_KEY",
)

# Load and index tax documents
docs = reader.load_data(["1040.pdf", "w2.pdf", "schedule_c.pdf"])
index = VectorStoreIndex.from_documents(docs)

# Query across all forms
query_engine = index.as_query_engine()
response = query_engine.query("What is the adjusted gross income?")
print(response)
# → "The adjusted gross income reported on Form 1040 is $83,200."

response = query_engine.query("How much federal tax was withheld?")
print(response)
# → "Federal income tax withheld as shown on the W-2 is $13,200."
```

---

## Key Features

### Concurrency Gate
A shared `asyncio.Semaphore` limits concurrent Azure DI calls so parallel
extractions never trigger 429 rate-limit responses.

```
Documents submitted         Azure DI calls in flight
  doc-1 ──┐                 ┌── slot 1
  doc-2 ──┤  Semaphore(12)  ├── slot 2
  doc-3 ──┤  ─────────────  ├── slot 3
  ...     │  max 12 at once │  ...
  doc-20 ─┘                 └── queued until slot free
```

Tune `max_concurrent` to your tier:
- F0 free tier → `max_concurrent=1`
- S0 paid tier → `max_concurrent=12` (safe empirically)

### 4-Stage Recovery Chain
Every document goes through a recovery chain before accepting an empty result:

| Stage | What it does | When triggered |
|-------|-------------|----------------|
| **Stage 0** | Direct Azure DI call on original bytes | Always first |
| **Stage 1** | Split into page chunks, analyse in parallel | Stage 0 empty or oversize |
| **Stage 2** | Re-render at 300 DPI (rasterise) | Stage 1 empty |
| **Stage 3** | Rotation block: as-is → 90° → 180° → 270° | After Stage 2 |

### 429 Retry Back-off
Exponential back-off with ±20% jitter on Azure DI rate limit responses.
Honors `Retry-After` header when present.

```
attempt 1 → wait  1s ± 200ms
attempt 2 → wait  2s ± 400ms
attempt 3 → wait  4s ± 800ms
attempt 4 → wait  8s ± 1.6s
attempt 5 → wait 16s ± 3.2s  (or propagate)
```

### Field Normalisation
Corrects known Azure DI output quirks automatically:

| Raw key from Azure DI | Normalised |
|---|---|
| `"Wages/Salary/Tips - HHA "` | `"Wages/Salary/Tips - HHA"` (trailing space) |
| `"SeconD Read"` | `"Second Read"` (typo) |
| `"Based Year> Tuition Paid"` | `"Based Year - Tuition Paid"` (`>` separator) |
| `'"75000"'` (quoted numeric) | `"75000"` (unquoted) |

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_concurrent` | `12` | Max simultaneous Azure DI calls |
| `pages_per_chunk` | `10` | Pages per chunk in Stage 1 split |
| `poll_timeout_seconds` | `120` | Per-call Azure DI timeout |
| `rate_limit_max_retries` | `5` | Max 429 retry attempts |
| `rate_limit_initial_delay_ms` | `1000` | Initial back-off delay (ms) |
| `rate_limit_max_delay_ms` | `32000` | Maximum back-off delay (ms) |
| `enable_audit_log` | `True` | Write extraction audit to file |
| `audit_log_dir` | `"logs"` | Directory for audit log |

---

## FERPA / PII Compliance

This library processes documents that **may contain sensitive taxpayer information**.
The following safeguards are built in:

| Concern | Safeguard |
|---------|-----------|
| Audit logs contain document names | Written to **file only** — never stdout or console |
| Credentials in CI | Stored as **GitHub Secrets**, never in code or logs |
| Sample data in repo | All samples use **fictional data** — no real SSNs or names |
| PDF files in repo | `samples/*.pdf` is **gitignored** — no documents committed |
| Log content | Only `document_id`, `kv_count`, `stage`, timing — **no field values logged** |

**Caller responsibility:** The extracted `KvEntry.value` fields may contain
SSNs, income figures, and other PII. Handle them according to your organisation's
data governance policy (FERPA, GLBA, or applicable regulations).

```python
# Example: strip SSN fields before indexing
docs = reader.load_data("1040.pdf")
for doc in docs:
    doc.text = "\n".join(
        line for line in doc.text.split("\n")
        if "social security" not in line.lower()
    )
```

---

## Azure Setup

1. Create an **Azure Document Intelligence** resource (S0 paid tier recommended)
2. Copy the **endpoint URL** and **API key** from Azure portal → Keys and Endpoint
3. The `prebuilt-document` model is used by default — no custom training required

---

## Development

```bash
git clone https://github.com/zavera/llama-index-readers-azure-tax-forms
cd llama-index-readers-azure-tax-forms
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run unit tests (no credentials needed — Azure DI is mocked)
pytest tests/ -v

# Generate fake sample PDFs for manual testing
pip install reportlab
cd samples && python generate_samples.py
```

---

## License

MIT License — Copyright (c) 2026 [Callisto Tech](https://github.com/zavera) / Ambreen Zaver
