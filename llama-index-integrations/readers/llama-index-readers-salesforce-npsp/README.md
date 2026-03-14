# LlamaIndex Reader: Salesforce NPSP

Connect Salesforce Nonprofit Success Pack (NPSP) donor data to LlamaIndex
RAG pipelines — no Airbyte, no CDK, no boilerplate.

## Overview

This reader ingests Salesforce **Contact** (donor) records, **Opportunity**
(gift) histories, and NPSP engagement metrics directly via `simple_salesforce`,
returning one LlamaIndex `Document` per donor with structured narrative text
an LLM can reason over directly.

Unlike `llama-index-readers-airbyte-salesforce` (raw JSON via Airbyte CDK),
this reader is **NPSP-aware**: it understands `npo02__` and `npsp__` field
prefixes and produces human-readable donor summaries — not raw JSON blobs.

## Installation

```bash
pip install llama-index-readers-salesforce-npsp
```

## Credentials

Set environment variables:

```bash
export SF_USERNAME="you@yourorg.org"
export SF_PASSWORD="your_password"
export SF_TOKEN="your_security_token"
```

## Quick Start

```python
from llama_index.readers.salesforce_npsp import SalesforceNPSPReader
from llama_index.core import VectorStoreIndex

reader = SalesforceNPSPReader(domain="login")
docs = reader.load_data(
    soql_filter="npo02__TotalOppAmount__c > 5000",
    limit=500,
)
index = VectorStoreIndex.from_documents(docs)
engine = index.as_query_engine()

response = engine.query(
    "Which major gift prospects haven't been contacted in 6 months?"
)
print(response)
```

## With PhilanthroPy Affinity Scores

Inject ML propensity scores from the PhilanthroPy sklearn toolkit
(https://github.com/PhilanthroPy-Project/PhilanthroPy):

```python
from philanthropy.models import DonorPropensityModel
import numpy as np

model = DonorPropensityModel(n_estimators=300, class_weight="balanced")
model.fit(X_train, y_train)


def affinity_scorer(meta):
    X = np.array([[meta["total_gift_amount"], meta["gift_count"], 0]])
    return model.predict_affinity_score(X)[0]


reader = SalesforceNPSPReader(
    domain="login", affinity_score_fn=affinity_scorer
)
docs = reader.load_data(limit=1000)
```

## API Reference

### SalesforceNPSPReader

| Parameter             | Type     | Default         | Description                          |
| --------------------- | -------- | --------------- | ------------------------------------ |
| username              | str      | SF_USERNAME env | Salesforce username                  |
| password              | str      | SF_PASSWORD env | Salesforce password                  |
| security_token        | str      | SF_TOKEN env    | Salesforce security token            |
| domain                | str      | "login"         | "login" for prod, "test" for sandbox |
| include_opportunities | bool     | True            | Fetch full gift history per donor    |
| affinity_score_fn     | Callable | None            | Optional scorer: (metadata) → float  |

### load_data()

| Parameter   | Type      | Default                        | Description                  |
| ----------- | --------- | ------------------------------ | ---------------------------- |
| contact_ids | List[str] | None                           | Specific Contact IDs to load |
| soql_filter | str       | "npo02**TotalOppAmount**c > 0" | SOQL WHERE clause            |
| limit       | int       | 500                            | Max records to return        |

Each returned Document has:

- `.text`: Human-readable donor narrative (name, giving summary, gift history)
- `.metadata`: donor_id, total_gift_amount, gift_count, last_gift_date,
  last_activity_date, affiliation, soft_credit_total, affinity_score (optional)
