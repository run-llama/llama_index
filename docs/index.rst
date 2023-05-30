.. GPT Index documentation master file, created by
   sphinx-quickstart on Sun Dec 11 14:30:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LlamaIndex 🦙 !
=====================================

LlamaIndex (GPT Index) is a data framework for your LLM application.

- Github: https://github.com/jerryjliu/llama_index
- PyPi:
   - LlamaIndex: https://pypi.org/project/llama-index/.
   - GPT Index (duplicate): https://pypi.org/project/gpt-index/.
- Twitter: https://twitter.com/llama_index
- Discord https://discord.gg/dGcwcsnxhU

Ecosystem
^^^^^^^^^

- 🏡 LlamaHub: https://llamahub.ai
- 🧪 LlamaLab: https://github.com/run-llama/llama-lab


🚀 Overview
-----------

Context
^^^^^^^
- LLMs are a phenomenonal piece of technology for knowledge generation and reasoning. They are pre-trained on large amounts of publicly available data.
- How do we best augment LLMs with our own private data?

We need a comprehensive toolkit to help perform this data augmentation for LLMs.


Proposed Solution
^^^^^^^^^^^^^^^^^
That's where **LlamaIndex** comes in. LlamaIndex is a "data framework" to help you build LLM apps. It provides the following tools:

- Offers **data connectors** to ingest your existing data sources and data formats (APIs, PDFs, docs, SQL, etc.)
- Provides ways to **structure your data** (indices, graphs) so that this data can be easily used with LLMs.
- Provides an **advanced retrieval/query interface over your data**: Feed in any LLM input prompt, get back retrieved context and knowledge-augmented output.
- Allows easy integrations with your outer application framework (e.g. with LangChain, Flask, Docker, ChatGPT, anything else).

LlamaIndex provides tools for both beginner users and advanced users. Our high-level API allows beginner users to use LlamaIndex to ingest and query their data in
5 lines of code. Our lower-level APIs allow advanced users to customize and extend any module (data connectors, indices, retrievers, query engines, reranking modules),
to fit their needs.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/installation.md
   getting_started/starter_example.md


.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   guides/primer.rst
   guides/tutorials.rst


.. toctree::
   :maxdepth: 2
   :caption: Use Cases
   :hidden:

   use_cases/queries.md
   use_cases/agents.md
   use_cases/apps.md


.. toctree::
   :maxdepth: 1
   :caption: Key Components
   :hidden:

   how_to/data_connectors.md
   how_to/indices.rst
   how_to/query_interface.rst
   how_to/customization.rst
   how_to/analysis.rst
   how_to/output_parsing.md
   how_to/evaluation/evaluation.md
   how_to/integrations.rst
   how_to/callbacks.rst
   how_to/storage.rst


.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   reference/indices.rst
   reference/query.rst
   reference/node.rst
   reference/llm_predictor.rst
   reference/node_postprocessor.rst
   reference/storage.rst
   reference/composability.rst
   reference/readers.rst
   reference/prompts.rst
   reference/service_context.rst
   reference/optimizers.rst
   reference/callbacks.rst
   reference/struct_store.rst
   reference/response.rst
   reference/playground.rst
   reference/node_parser.rst
   reference/example_notebooks.rst
   reference/langchain_integrations/base.rst


.. toctree::
   :maxdepth: 1
   :caption: Gallery
   :hidden:

   gallery/app_showcase.md
