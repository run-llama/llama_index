<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

# Welcome to LlamaIndex 🦙 !

LlamaIndex is a data framework for [LLM](https://en.wikipedia.org/wiki/Large_language_model)-based applications which benefit from context augmentation. Such LLM systems have been termed as RAG systems, standing for "Retrieval-Augemented Generation". LlamaIndex provides the essential abstractions to more easily ingest, structure, and access private or domain-specific data in order to inject these safely and reliably into LLMs for more accurate text generation. It's available in Python (these docs) and [Typescript](https://ts.llamaindex.ai/).

!!! tip

    Updating to LlamaIndex v0.10.0? Check out the [migration guide](./getting_started/v0_10_0_migration.md).

## 🚀 Why Context Augmentation?

LLMs offer a natural language interface between humans and data. Widely available models come pre-trained on huge amounts of publicly available data like Wikipedia, mailing lists, textbooks, source code and more.

However, while LLMs are trained on a great deal of data, they are not trained on **your** data, which may be private or specific to the problem you're trying to solve. It's behind APIs, in SQL databases, or trapped in PDFs and slide decks.

You may choose to **fine-tune** a LLM with your data, but:

- Training a LLM is **expensive**.
- Due to the cost to train, it's **hard to update** a LLM with latest information.
- **Observability** is lacking. When you ask a LLM a question, it's not obvious how the LLM arrived at its answer.

Instead of fine-tuning, one can a context augmentation pattern called [Retrieval-Augmented Generation (RAG)](./getting_started/concepts.md) to obtain more accurate text generation relevant to your specific data. RAG involves the following high level steps:

1. Retrieve information from your data sources first,
2. Add it to your question as context, and
3. Ask the LLM to answer based on the enriched prompt.

In doing so, RAG overcomes all three weaknesses of the fine-tuning approach:

- There's no training involved, so it's **cheap**.
- Data is fetched only when you ask for them, so it's **always up to date**.
- LlamaIndex can show you the retrieved documents, so it's **more trustworthy**.

### 🦙 Why LlamaIndex for Context Augmentation?

Firstly, LlamaIndex imposes no restriction on how you use LLMs. You can still use LLMs as auto-complete, chatbots, semi-autonomous agents, and more (see Use Cases on the left). It only makes LLMs more relevant to you.

LlamaIndex provides the following tools to help you quickly standup production-ready RAG systems:

- **Data connectors** ingest your existing data from their native source and format. These could be APIs, PDFs, SQL, and (much) more.
- **Data indexes** structure your data in intermediate representations that are easy and performant for LLMs to consume.
- **Engines** provide natural language access to your data. For example:

  - Query engines are powerful retrieval interfaces for knowledge-augmented output.
  - Chat engines are conversational interfaces for multi-message, "back and forth" interactions with your data.

- **Data agents** are LLM-powered knowledge workers augmented by tools, from simple helper functions to API integrations and more.
- **Application integrations** tie LlamaIndex back into the rest of your ecosystem. This could be LangChain, Flask, Docker, ChatGPT, or… anything else!

### 👨‍👩‍👧‍👦 Who is LlamaIndex for?

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
- npm (Typescript/Javascript):
  - [LlamaIndex.TS Github](https://github.com/run-llama/LlamaIndexTS)
  - [TypeScript Docs](https://ts.llamaindex.ai/)
  - [LlamaIndex.TS](https://www.npmjs.com/package/llamaindex)

## Community

Need help? Have a feature suggestion? Join the LlamaIndex community:

- [Twitter](https://twitter.com/llama_index)
- [Discord](https://discord.gg/dGcwcsnxhU)

## Associated projects

- [🏡 LlamaHub](https://llamahub.ai) | A large (and growing!) collection of custom data connectors
- [SEC Insights](https://sec-insights.com) | A LlamaIndex-powered application for financial research
- [create-llama](https://www.npmjs.com/package/create-llama) | A CLI tool to quickly scaffold LlamaIndex projects
