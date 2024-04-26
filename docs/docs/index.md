<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

# Welcome to LlamaIndex 🦙 !

LlamaIndex is a framework for building **context-augmented** [LLM](https://en.wikipedia.org/wiki/Large_language_model) applications. Context augmentation refers to any use case that applies LLMs on top of your private or domain-specific data. Some popular [use cases](./use_cases/index.md) include the following:

- Question-Answering Chatbots (commonly referred to as RAG systems, which stands for "Retrieval-Augmented Generation")
- Document Understanding and Extraction
- Autonomous Agents that can perform research and take actions


LlamaIndex provides the tools to build any of these above use cases from prototype to production. The tools allow you to both ingest/process this data and implement complex query workflows combining data access with LLM prompting.

LlamaIndex is available in Python (these docs) and [Typescript](https://ts.llamaindex.ai/).

!!! tip
    Updating to LlamaIndex v0.10.0? Check out the [migration guide](./getting_started/v0_10_0_migration.md).

## 🚀 Why Context Augmentation?

LLMs offer a natural language interface between humans and data. Widely available models come pre-trained on huge amounts of publicly available data. However, they are not trained on **your** data, which may be private or specific to the problem you're trying to solve. It's behind APIs, in SQL databases, or trapped in PDFs and slide decks.

LlamaIndex provides tooling to enable context augmentation. A popular example is [Retrieval-Augmented Generation (RAG)](./getting_started/concepts.md) which combines context with LLMs at inference time. Another is [finetuning](./use_cases/fine_tuning.md).

## 🦙 LlamaIndex is the Data Framework for Context-Augmented LLM Apps

LlamaIndex imposes no restriction on how you use LLMs. You can still use LLMs as auto-complete, chatbots, semi-autonomous agents, and more. It only makes LLMs more relevant to you.

LlamaIndex provides the following tools to help you quickly standup production-ready LLM applications:

- **Data connectors** ingest your existing data from their native source and format. These could be APIs, PDFs, SQL, and (much) more.
- **Data indexes** structure your data in intermediate representations that are easy and performant for LLMs to consume.
- **Engines** provide natural language access to your data. For example:
    - Query engines are powerful interfaces for question-answering (e.g. a RAG pipeline).
    - Chat engines are conversational interfaces for multi-message, "back and forth" interactions with your data.
- **Agents** are LLM-powered knowledge workers augmented by tools, from simple helper functions to API integrations and more.
- **Observability/Evaluation** integrations that enable you to rigorously experiment, evaluate, and monitor your app in a virtuous cycle.

## 👨‍👩‍👧‍👦 Who is LlamaIndex for?

LlamaIndex provides tools for beginners, advanced users, and everyone in between.

Our high-level API allows beginner users to use LlamaIndex to ingest and query their data in 5 lines of code.

For more complex applications, our lower-level APIs allow advanced users to customize and extend any module—data connectors, indices, retrievers, query engines, reranking modules—to fit their needs.

## Getting Started

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
