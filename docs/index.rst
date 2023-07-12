.. LlamaIndex documentation master file, created by
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
   getting_started/concepts.md
   getting_started/customization.rst

.. toctree::
   :maxdepth: 2
   :caption: End-to-End Tutorials
   :hidden:

   end_to_end_tutorials/usage_pattern.md
   end_to_end_tutorials/discover_llamaindex.md
   end_to_end_tutorials/use_cases.md
   
.. toctree::
   :maxdepth: 1
   :caption: Index/Data Modules
   :hidden:

   core_modules/data_modules/connector/root.md
   core_modules/data_modules/documents_and_nodes/root.md
   core_modules/data_modules/node_parsers/root.md
   core_modules/data_modules/storage/root.md
   core_modules/data_modules/index/root.md

.. toctree::
   :maxdepth: 1
   :caption: Query Modules
   :hidden:

   core_modules/query_modules/retriever/root.md
   core_modules/query_modules/node_postprocessors/root.md
   core_modules/query_modules/response_synthesizers/root.md
   core_modules/query_modules/structured_outputs/root.md
   core_modules/query_modules/query_engine/root.md
   core_modules/query_modules/chat_engines/root.md

.. toctree::
   :maxdepth: 1
   :caption: Agent Modules
   :hidden:

   core_modules/agent_modules/agents/root.md

.. toctree::
   :maxdepth: 1
   :caption: Model Modules
   :hidden:

   core_modules/model_modules/llms/root.md
   core_modules/model_modules/embeddings/root.md
   core_modules/model_modules/prompts.md

.. toctree::
   :maxdepth: 1
   :caption: Supporting Modules

   core_modules/supporting_modules/service_context.md
   core_modules/supporting_modules/callbacks/root.md
   core_modules/supporting_modules/evaluation/root.md
   core_modules/supporting_modules/cost_analysis/root.md
   core_modules/supporting_modules/playground/root.md

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   development/contributing.rst
   development/documentation.rst
   development/privacy.md
   development/changelog.rst

.. toctree::
   :maxdepth: 2
   :caption: Community
   :hidden:

   community/integrations.md
   community/app_showcase.md

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api_reference/index.rst
