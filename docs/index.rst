.. GPT Index documentation master file, created by
   sphinx-quickstart on Sun Dec 11 14:30:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LlamaIndex 🦙 (GPT Index)!
=====================================

LlamaIndex (GPT Index) is a project that provides a central interface to connect your LLM's with external data.

- Github: https://github.com/jerryjliu/llama_index
- PyPi:
   - LlamaIndex: https://pypi.org/project/llama-index/.
   - GPT Index (duplicate): https://pypi.org/project/gpt-index/.
- Twitter: https://twitter.com/gpt_index
- Discord https://discord.gg/dGcwcsnxhU


🚀 Overview
-----------

Context
^^^^^^^
- LLMs are a phenomenonal piece of technology for knowledge generation and reasoning. They are pre-trained on large amounts of publicly available data.
- How do we best augment LLMs with our own private data?
- One paradigm that has emerged is *in-context* learning (the other is finetuning), where we insert context into the input prompt. That way, we take advantage of the LLM's reasoning capabilities to generate a response.

To perform LLM's data augmentation in a performant, efficient, and cheap manner, we need to solve two components:

- Data Ingestion
- Data Indexing

Proposed Solution
^^^^^^^^^^^^^^^^^
That's where the **LlamaIndex** comes in. LlamaIndex is a simple, flexible interface between your external data and LLMs. It provides the following tools in an easy-to-use fashion:

- Offers `data connectors <http://llamahub.ai>`_ to your existing data sources and data formats (API's, PDF's, docs, SQL, etc.)
- Provides **indices** over your unstructured and structured data for use with LLM's. These indices help to abstract away common boilerplate and pain points for in-context learning:

   - Storing context in an easy-to-access format for prompt insertion.
   - Dealing with prompt limitations (e.g. 4096 tokens for Davinci) when context is too big.
   - Dealing with text splitting.
- Provides users an interface to **query** the index (feed in an input prompt) and obtain a knowledge-augmented output.
- Offers you a comprehensive toolset trading off cost and performance.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation.md
   getting_started/starter_example.md


.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/primer.rst
   guides/tutorials.rst
   guides/notebooks.rst


.. toctree::
   :maxdepth: 2
   :caption: Use Cases

   use_cases/queries.md
   use_cases/apps.md


.. toctree::
   :maxdepth: 1
   :caption: Key Components

   how_to/data_connectors.md
   how_to/indices.rst
   how_to/query_interface.rst
   how_to/customization.rst
   how_to/analysis.rst
   how_to/output_parsing.md
   how_to/integrations.rst

   .. evaluation
   .. integrations

   .. how_to/composability.md
   .. how_to/update.md
   .. how_to/cost_analysis.md
   .. how_to/vector_stores.md
   .. how_to/using_with_langchain.md


.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/indices.rst
   reference/query.rst
   reference/node.rst
   reference/docstore.rst
   reference/composability.rst
   reference/readers.rst
   reference/prompts.rst
   reference/service_context.rst
   reference/optimizers.rst
   reference/struct_store.rst
   reference/response.rst
   reference/playground.rst
   reference/node_parser.rst
   reference/example_notebooks.rst
   reference/langchain_integrations/base.rst


.. toctree::
   :maxdepth: 1
   :caption: Gallery

   gallery/app_showcase.md
