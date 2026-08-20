# LlamaIndex Retrievers Integration: Bedrock

## Knowledge Bases

> [Knowledge bases for Amazon Bedrock](https://aws.amazon.com/bedrock/knowledge-bases/) is an Amazon Web Services (AWS) offering which lets you quickly build RAG applications by using your private data to customize FM response.

> Implementing `RAG` requires organizations to perform several cumbersome steps to convert data into embeddings (vectors), store the embeddings in a specialized vector database, and build custom integrations into the database to search and retrieve text relevant to the user's query. This can be time-consuming and inefficient.

> With `Knowledge Bases for Amazon Bedrock`, simply point to the location of your data in `Amazon S3`, and `Knowledge Bases for Amazon Bedrock` takes care of the entire ingestion workflow into your vector database. If you do not have an existing vector database, Amazon Bedrock creates an Amazon OpenSearch Serverless vector store for you.

> Knowledge base can be configured through [AWS Console](https://aws.amazon.com/console/) or by using [AWS SDKs](https://aws.amazon.com/developer/tools/).

## Installation

```
pip install llama-index-retrievers-bedrock
```

## Usage

### Managed Knowledge Base (Recommended)

Managed knowledge bases let Bedrock handle embedding, storage, and retrieval automatically — no external vector store required:

```python
from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever

# Managed KB (pass knowledge_base_type="MANAGED"):
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="<knowledge-base-id>",
    retrieval_config={
        "managedSearchConfiguration": {
            "numberOfResults": 4,
        }
    },
)

query = "How big is Milky Way as compared to the entire universe?"
retrieved_results = retriever.retrieve(query)
print(retrieved_results[0].get_content())
```

### Vector Knowledge Base (Legacy)

Traditional vector knowledge bases with explicit embedding and vector store configuration:

```python
from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="<knowledge-base-id>",
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 4,
            "overrideSearchType": "HYBRID",
            "filter": {"equals": {"key": "tag", "value": "space"}},
        }
    },
)

query = "How big is Milky Way as compared to the entire universe?"
retrieved_results = retriever.retrieve(query)
print(retrieved_results[0].get_content())
```

> **SDK requirements:** `boto3 >= 1.43` for managed search and agentic retrieval.

**Reranking options** for managed search: `MANAGED` (default — automatic), `NONE` (disable reranking), `CUSTOM` (your own Bedrock reranking model e.g. Cohere Rerank v3.5).

**Required IAM Permissions:**
```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:Retrieve",
    "bedrock:AgenticRetrieveStream"
  ],
  "Resource": "arn:aws:bedrock:<region>:<account-id>:knowledge-base/<kb-id>"
}
```

**Resources:** [Build a Managed KB](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-build-managed.html) | [Retrieve API](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-retrieve.html) | [Agentic Retrieval](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-agentic.html)

## Notebook

Explore the retriever using Notebook present at:
https://docs.llamaindex.ai/en/latest/examples/retrievers/bedrock_retriever/
