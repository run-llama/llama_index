.. LlamaIndex documentation master file, created by
   sphinx-quickstart on Sun Dec 11 14:30:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LlamaIndex 🦙 !
##########################

LlamaIndex (formerly GPT Index) is a data framework for LLM applications to ingest, structure, and access private or domain-specific data.

🚀 Why LlamaIndex?
******************

At their core, LLMs offer a natural language interface between humans and inferred data. Widely available models come pre-trained on huge amounts of publicly available data, from Wikipedia and mailing lists to textbooks and source code.

Applications built on top of LLMs often require augmenting these models with private or domain-specific data. Unfortunately, that data can be distributed across siloed applications and data stores. It's behind APIs, in SQL databases, or trapped in PDFs and slide decks.

That's where **LlamaIndex** comes in.

🦙 How can LlamaIndex help?
***************************

LlamaIndex provides the following tools:

- **Data connectors** ingest your existing data from their native source and format. These could be APIs, PDFs, SQL, and (much) more.
- **Data indexes** structure your data in intermediate representations that are easy and performant for LLMs to consume.
- **Engines** provide natural language access to your data. For example:

  - Query engines are powerful retrieval interfaces for knowledge-augmented output.
  - Chat engines are conversational interfaces for multi-message, "back and forth" interactions with your data.
- **Application integrations** tie LlamaIndex back into the rest of your ecosystem. This could be LangChain, Flask, Docker, ChatGPT, or… anything else!

👨‍👩‍👧‍👦 Who is LlamaIndex for?
*************************

LlamaIndex provides tools for beginners, advanced users, and everyone in between.

Our high-level API allows beginner users to use LlamaIndex to ingest and query their data in 5 lines of code.

For more complex applications, our lower-level APIs allow advanced users to customize and extend any module—data connectors, indices, retrievers, query engines, reranking modules—to fit their needs.

🗺️ Ecosystem
************

To download or contribute, find LlamaIndex on:

- Github: https://github.com/jerryjliu/llama_index
- PyPi:

  - LlamaIndex: https://pypi.org/project/llama-index/.
  - GPT Index (duplicate): https://pypi.org/project/gpt-index/.

Community
---------
Need help? Have a feature suggestion? Join the LlamaIndex community:

- Twitter: https://twitter.com/llama_index
- Discord https://discord.gg/dGcwcsnxhU

Associated projects
-------------------

- 🏡 LlamaHub: https://llamahub.ai | A large (and growing!) collection of custom data connectors
- 🧪 LlamaLab: https://github.com/run-llama/llama-lab | Ambitious projects built on top of LlamaIndex

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
   core_modules/agent_modules/tools/root.md

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
   :hidden:

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

.. toctree::
   :maxdepth: 1
   :hidden:

   deprecated_terms.md
