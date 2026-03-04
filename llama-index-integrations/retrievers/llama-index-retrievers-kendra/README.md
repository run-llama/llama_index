# LlamaIndex Retrievers Integration: Amazon Kendra

## Overview

> [Amazon Kendra](https://aws.amazon.com/kendra/) is an intelligent search service powered by machine learning. Kendra reimagines enterprise search for your websites and applications by allowing users to search your unstructured and structured data using natural language.

> Kendra supports a wide variety of data sources including:
>
> - Documents (PDF, Word, PowerPoint, HTML)
> - FAQs
> - Knowledge bases
> - Databases
> - Websites
> - Custom data sources through connectors

## Installation

```bash
pip install llama-index-retrievers-kendra
```

## Usage

Here's a basic example of how to use the Kendra retriever:

```python
from llama_index.retrievers.kendra import AmazonKendraRetriever

retriever = AmazonKendraRetriever(
    index_id="<kendra-index-id>",
    query_config={
        "PageSize": 4,
        "AttributeFilter": {
            "EqualsTo": {
                "Key": "department",
                "Value": {"StringValue": "engineering"},
            }
        },
    },
)

query = "What is our company's remote work policy?"
retrieved_results = retriever.retrieve(query)

# Print the first retrieved result
print(retrieved_results[0].get_content())
```

## Advanced Configuration

The retriever supports Kendra's rich querying capabilities through the `query_config` parameter:

```python
retriever = AmazonKendraRetriever(
    index_id="<kendra-index-id>",
    query_config={
        "PageSize": 10,  # Number of results to return
        "AttributeFilter": {
            # Filter results based on document attributes
            "AndAllFilters": [
                {
                    "EqualsTo": {
                        "Key": "department",
                        "Value": {"StringValue": "engineering"},
                    }
                },
                {
                    "GreaterThan": {
                        "Key": "last_updated",
                        "Value": {"StringValue": "2023-01-01"},
                    }
                },
            ]
        },
        "QueryResultTypeFilter": "DOCUMENT",  # Only return document results
    },
)
```

## Confidence Scores

The retriever maps Kendra's confidence levels to float scores as follows:

- VERY_HIGH: 1.0
- HIGH: 0.8
- MEDIUM: 0.6
- LOW: 0.4
- NOT_AVAILABLE: 0.0

These scores can be accessed through the `score` attribute of the retrieved nodes:

```python
results = retriever.retrieve("query")
for result in results:
    print(f"Text: {result.get_content()}")
    print(f"Confidence Score: {result.score}")
```

## Authentication

The retriever supports various AWS authentication methods:

```python
retriever = AmazonKendraRetriever(
    index_id="<kendra-index-id>",
    profile_name="my-aws-profile",  # Use AWS profile
    region_name="us-west-2",  # Specify AWS region
    # Or use explicit credentials
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    aws_session_token="YOUR_SESSION_TOKEN",  # Optional
)
```
