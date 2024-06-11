<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

# Welcome to LlamaIndex 🦙 !

LlamaIndex is a framework for building context-augmented generative AI applications with [LLMs](https://en.wikipedia.org/wiki/Large_language_model).

<div class="grid cards" markdown>

-  <span style="font-size: 200%">[Introduction](#introduction)</span>

    What is context augmentation? How does LlamaIndex help?

-   <span style="font-size: 200%">[Use cases](#use-cases)</span>

    What kind of apps can you build with LlamaIndex? Who should use it?

-   <span style="font-size: 200%">[Getting started](#getting-started)</span>

    Get started in Python or TypeScript in just 5 lines of code!

-   <span style="font-size: 200%">[LlamaCloud](#llamacloud)</span>

    Managed services for LlamaIndex including [LlamaParse](https://docs.cloud.llamaindex.ai/llamaparse/getting_started), the world's best document parser.

-   <span style="font-size: 200%">[Community](#community)</span>

    Get help and meet collaborators on Discord, Twitter and more.

-   <span style="font-size: 200%">[Related projects](#related-projects)</span>

    Check out our library of connectors at [LlamaHub](https://llamahub.ai), demos and more.

</div>

## Introduction

### What is context augmentation?

LLMs offer a natural language interface between humans and data. LLMs come pre-trained on huge amounts of publicly available data, but they are not trained on **your** data. Your data may be private or specific to the problem you're trying to solve. It's behind APIs, in SQL databases, or trapped in PDFs and slide decks.

Context augmentation makes your data available to the LLM to solve the problem at hand. LlamaIndex provides the tools to build any of context-augmentation use case, from prototype to production. Our tools allow you to ingest, parse, index and process your data and quickly implement complex query workflows combining data access with LLM prompting.

The most popular example of context-augmentation is [Retrieval-Augmented Generation or RAG](./getting_started/concepts.md), which combines context with LLMs at inference time.

### LlamaIndex is the Data Framework for Context-Augmented LLM Apps

LlamaIndex imposes no restriction on how you use LLMs. You can use LLMs as auto-complete, chatbots, semi-autonomous agents, and more. It just makes using them easier. We provide tools like:

- **Data connectors** ingest your existing data from their native source and format. These could be APIs, PDFs, SQL, and (much) more.
- **Data indexes** structure your data in intermediate representations that are easy and performant for LLMs to consume.
- **Engines** provide natural language access to your data. For example:
    - Query engines are powerful interfaces for question-answering (e.g. a RAG pipeline).
    - Chat engines are conversational interfaces for multi-message, "back and forth" interactions with your data.
- **Agents** are LLM-powered knowledge workers augmented by tools, from simple helper functions to API integrations and more.
- **Observability/Evaluation** integrations that enable you to rigorously experiment, evaluate, and monitor your app in a virtuous cycle.

## Use cases

Some popular use cases include:

- [Question-Answering Chatbots](/use_cases/q_and_a/) (Retrieval-Augmented Generation aka RAG)
- Document Understanding and Extraction
- Autonomous Agents that can perform research and take actions

Check out our [use cases](./use_cases/index.md) documentation for more examples and links to tutorials.

### 👨‍👩‍👧‍👦 Who is LlamaIndex for?

LlamaIndex provides tools for beginners, advanced users, and everyone in between.

Our high-level API allows beginner users to use LlamaIndex to ingest and query their data in 5 lines of code.

For more complex applications, our lower-level APIs allow advanced users to customize and extend any module—data connectors, indices, retrievers, query engines, reranking modules—to fit their needs.

## Getting Started

LlamaIndex is available in Python (these docs) and [Typescript](https://ts.llamaindex.ai/).

To install the library:

`pip install llama-index`

We recommend starting at [how to read these docs](./getting_started/reading.md) which will point you to the right place based on your experience level.

## 🗺️ Ecosystem

To download or contribute, find LlamaIndex on:

- [Github](https://github.com/run-llama/llama_index)
- [PyPi](https://pypi.org/project/llama-index/)
- LlamaIndex.TS (Typescript/Javascript package):
    - [LlamaIndex.TS Github](https://github.com/run-llama/LlamaIndexTS)
    - [TypeScript Docs](https://ts.llamaindex.ai/)
    - [LlamaIndex.TS npm](https://www.npmjs.com/package/llamaindex)

## LlamaCloud

If you're an enterprise developer, check out [**LlamaCloud**](https://www.llamaindex.ai/enterprise). It is a managed platform for data parsing and ingestion, allowing
you to get production-quality data for your production LLM application.

Check out the following resources:

- [**LlamaParse**](./llama_cloud/llama_parse.md): our state-of-the-art document parsing solution. Part of LlamaCloud and also available as a self-serve API. [Signup here for API access](https://cloud.llamaindex.ai/).
- [**LlamaCloud**](./llama_cloud/index.md): our e2e data platform. In private preview with startup and enterprise plans. [Talk to us](https://www.llamaindex.ai/contact) if interested.

## Community

Need help? Have a feature suggestion? Join the LlamaIndex community:

- [Twitter](https://twitter.com/llama_index)
- [Discord](https://discord.gg/dGcwcsnxhU)

## Associated projects

- [🏡 LlamaHub](https://llamahub.ai) | A large (and growing!) collection of custom data connectors
- [SEC Insights](https://secinsights.ai) | A LlamaIndex-powered application for financial research
- [create-llama](https://www.npmjs.com/package/create-llama) | A CLI tool to quickly scaffold LlamaIndex projects
