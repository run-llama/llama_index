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

## With Affinity Scores

Pass any callable `(metadata: dict) -> float` as `affinity_score_fn`.
The function is called once per donor at load time and the result is stored as
`metadata["affinity_score"]` on every returned `Document`.

```python
def affinity_scorer(meta: dict) -> float:
    """Simple heuristic: scale 0-100 from lifetime giving + gift frequency."""
    total = meta.get("total_gift_amount", 0) or 0
    count = meta.get("gift_count", 0) or 0
    return round(min(50 + (total / 5_000) + count * 2, 100), 1)


reader = SalesforceNPSPReader(
    domain="login",
    affinity_score_fn=affinity_scorer,
)
docs = reader.load_data(limit=1000)
# docs[0].metadata["affinity_score"]  →  float
```

In production, replace the body of `affinity_scorer` with a call to your
trained propensity model. The function receives the full metadata dictionary
so it has access to all donor fields listed in the API reference below.

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

| Parameter     | Type        | Default                          | Description                                                                          |
| ------------- | ----------- | -------------------------------- | ------------------------------------------------------------------------------------ |
| `contact_ids` | `List[str]` | `None`                           | Explicit Contact IDs to fetch. When provided, `soql_filter` and `limit` are ignored. |
| `soql_filter` | `str`       | `"npo02__TotalOppAmount__c > 0"` | SOQL `WHERE` clause applied to the `Contact` object.                                 |
| `limit`       | `int`       | `500`                            | Maximum number of donor records to return (ignored when `contact_ids` is set).       |

Each returned `Document` has:

- **`.text`** — human-readable donor narrative: name, giving summary, and up to 10 most-recent gifts.
- **`.metadata`** — structured dictionary of donor fields:

| Key                    | Type            | Description                                                           |
| ---------------------- | --------------- | --------------------------------------------------------------------- |
| `donor_id`             | `str`           | Salesforce Contact ID                                                 |
| `donor_name`           | `str`           | First + Last name                                                     |
| `email`                | `str`           | Contact email                                                         |
| `affiliation`          | `str`           | Primary affiliated organisation (`npsp__Primary_Affiliation__r.Name`) |
| `total_gift_amount`    | `float`         | Lifetime giving total (`npo02__TotalOppAmount__c`)                    |
| `gift_count`           | `int`           | Number of closed gifts (`npo02__NumberOfClosedOpps__c`)               |
| `average_gift_amount`  | `float`         | Average gift size (`npo02__AverageAmount__c`)                         |
| `largest_gift_amount`  | `float`         | Largest single gift (`npo02__LargestAmount__c`)                       |
| `first_gift_date`      | `str`           | Date of first gift (`npo02__FirstCloseDate__c`)                       |
| `last_gift_date`       | `str`           | Date of most recent gift (`npo02__LastCloseDate__c`)                  |
| `last_activity_date`   | `str`           | Last CRM activity date (`LastActivityDate`)                           |
| `soft_credit_total`    | `float`         | Total soft credits (`npsp__Soft_Credit_Total__c`)                     |
| `planned_giving_count` | `int`           | Number of planned gifts (`npsp__Planned_Giving_Count__c`)             |
| `source`               | `str`           | Always `"salesforce_npsp"`                                            |
| `affinity_score`       | `float \| None` | Set only when `affinity_score_fn` is provided                         |
