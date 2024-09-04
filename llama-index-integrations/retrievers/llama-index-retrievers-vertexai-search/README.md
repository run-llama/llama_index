# LlamaIndex Retrievers Integration: Vertex AI Search

## Vertex AI Search

> [Vertex AI Search](https://cloud.google.com/enterprise-search?e=48754805&hl=en) helps developers build secure, Google-quality search experiences for websites, intranet and RAG systems for generative AI agents and apps. Vertex AI Search is a part of Vertex AI Agent Builder.

> [Vertex AI Agent Builder](https://cloud.google.com/products/agent-builder?e=48754805&hl=en#build-and-deploy-enterprise-ready-generative-ai-experiences): Create AI agents and applications using natural language or a code-first approach. Easily ground your agents or apps in enterprise data with a range of options. Vertex AI Agent Builder gathers all the surfaces and tools that developers need to build their AI agents and applications..

## Installation

```
pip install llama-index-retrievers-vertexai-search
```

## Usage

```
from llama_index.retrievers.vertexai_search import VertexAISearchRetriever

retriever = VertexAISearchRetriever(
        project_id=PROJECT_ID,
        data_store_id=DATA_STORE_ID,
        location_id=LOCATION_ID,
        engine_data_type=1
    )

query = "harry potter"
retrieved_results = retriever.retrieve(query)

# Prints the first retrieved result
print(retrieved_results[0].get_content())
```

## Notebook

Explore the retriever using Notebook present at:
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/vertex_ai_search_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
