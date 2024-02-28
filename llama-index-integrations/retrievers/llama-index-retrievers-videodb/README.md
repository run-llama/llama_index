# VideoDB Retriever

## Overview

[VideoDB](https://videodb.io) is a serverless database designed to streamline the storage, search, editing, and streaming of video content. VideoDB offers random access to sequential video data by building indexes and developing interfaces for querying and browsing video content. Learn more at [docs.videodb.io](https://docs.videodb.io).

## Getting Started

### Prerequisites

- Obtain API keys from [VideoDB dashboard](https://console.videodb.io)

### Installation

Install the necessary packages with the following command:

```
pip install llama-index llama-index-retrievers-videodb videodb
```

## Building Your Pipeline

1. **Data Ingestion**: Upload your videos to VideoDB and leverage its managed indexing for efficient data organization, choosing between semantic or scene-based indexing.
2. **Querying**: Utilize `VideoDBRetriever` to retrieve relevant video segments and `llama-index` for constructing your RAG pipeline, enhancing your LLM's context with video-based insights.

## 👨‍👩‍👧‍👦 Support & Community

If you have any questions or feedback.
Please feel free to reach out to us

- [Discord](https://discord.gg/py9P639jGz)
- [Github](https://github.com/video-db)
- [VideoDB](https://videodb.io)
- [Email](mailto:ashu@videodb.io)
