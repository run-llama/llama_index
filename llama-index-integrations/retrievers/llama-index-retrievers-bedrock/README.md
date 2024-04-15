# LlamaIndex Retrievers Integration: Bedrock

## Knowledge Bases

> [Knowledge bases for Amazon Bedrock](https://aws.amazon.com/bedrock/knowledge-bases/) is an Amazon Web Services (AWS) offering which lets you quickly build RAG applications by using your private data to customize FM response.

> Implementing `RAG` requires organizations to perform several cumbersome steps to convert data into embeddings (vectors), store the embeddings in a specialized vector database, and build custom integrations into the database to search and retrieve text relevant to the user’s query. This can be time-consuming and inefficient.

> With `Knowledge Bases for Amazon Bedrock`, simply point to the location of your data in `Amazon S3`, and `Knowledge Bases for Amazon Bedrock` takes care of the entire ingestion workflow into your vector database. If you do not have an existing vector database, Amazon Bedrock creates an Amazon OpenSearch Serverless vector store for you.

> Knowledge base can be configured through [AWS Console](https://aws.amazon.com/console/) or by using [AWS SDKs](https://aws.amazon.com/developer/tools/).

### Notebook

Explore the retriever using Notebook present at:

```
docs/docs/examples/retrievers/bedrock_retriever.ipynb
```
