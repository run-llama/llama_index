Welcome to LlamaIndex 🦙 !
##########################

LlamaIndex (formerly GPT Index) is a data framework for `LLM <https://en.wikipedia.org/wiki/Large_language_model>`_-based applications to ingest, structure, and access private or domain-specific data. It's available in Python (these docs) and `Typescript <https://ts.llamaindex.ai/>`_.

🚀 Why LlamaIndex?
******************

LLMs offer a natural language interface between humans and data. Widely available models come pre-trained on huge amounts of publicly available data like Wikipedia, mailing lists, textbooks, source code and more.

However, while LLMs are trained on a great deal of data, they are not trained on **your** data, which may private or specific to the problem you're trying to solve. It's behind APIs, in SQL databases, or trapped in PDFs and slide decks.

LlamaIndex solves this problem by connecting to these data sources and adding your data to the data LLMs already have. This is often called Retrieval-Augmented Generation (RAG). RAG enables you to use LLMs to query your data, transform it, and generate new insights. You can ask questions about your data, create chatbots, build semi-autonomous agents, and more. To learn more, see our `Use Cases <./end_to_end_tutorials/use_cases.html>`_.

🦙 How can LlamaIndex help?
***************************

LlamaIndex provides the following tools:

- **Data connectors** ingest your existing data from their native source and format. These could be APIs, PDFs, SQL, and (much) more.
- **Data indexes** structure your data in intermediate representations that are easy and performant for LLMs to consume.
- **Engines** provide natural language access to your data. For example:
  - Query engines are powerful retrieval interfaces for knowledge-augmented output.
  - Chat engines are conversational interfaces for multi-message, "back and forth" interactions with your data.
- **Data agents** are LLM-powered knowledge workers augmented by tools, from simple helper functions to API integrations and more.
- **Application integrations** tie LlamaIndex back into the rest of your ecosystem. This could be LangChain, Flask, Docker, ChatGPT, or… anything else!

👨‍👩‍👧‍👦 Who is LlamaIndex for?
*************************

LlamaIndex provides tools for beginners, advanced users, and everyone in between.

Our high-level API allows beginner users to use LlamaIndex to ingest and query their data in 5 lines of code.

For more complex applications, our lower-level APIs allow advanced users to customize and extend any module—data connectors, indices, retrievers, query engines, reranking modules—to fit their needs.

Getting Started
****************
``pip install llama-index``

Our documentation includes detailed `Installation Instructions <./getting_started/installation.html>`_ and a `Starter Tutorial <./getting_started/starter_example.html>`_ to build your first application (in five lines of code!)

Once you're up and running, `High-Level Concepts <./getting_started/concepts.html>`_ has an overview of LlamaIndex's modular architecture. For more hands-on practical examples, look through our `End-to-End Tutorials <./end_to_end_tutorials/use_cases.html>`_ or learn how to `customize <./getting_started/customization.html>`_ components to fit your specific needs.

**NOTE**: We have a Typescript package too! `Repo <https://github.com/run-llama/LlamaIndexTS>`_, `Docs <https://ts.llamaindex.ai/>`_

🗺️ Ecosystem
************

To download or contribute, find LlamaIndex on:

- Github: https://github.com/jerryjliu/llama_index
- PyPi:

  - LlamaIndex: https://pypi.org/project/llama-index/.
  - GPT Index (duplicate): https://pypi.org/project/gpt-index/.

- NPM (Typescript/Javascript):
   - Github: https://github.com/run-llama/LlamaIndexTS
   - Docs: https://ts.llamaindex.ai/
   - LlamaIndex.TS: https://www.npmjs.com/package/llamaindex

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
   getting_started/discover_llamaindex.md

.. toctree::
   :maxdepth: 2
   :caption: Use Cases
   :hidden:

   use_cases/q_and_a.md
   use_cases/chatbots.md
   use_cases/agents.md

.. toctree::
   :maxdepth: 2
   :caption: Understanding
   :hidden:

   understanding/understanding.md
   understanding/using_llms/using_llms.md
   understanding/loading/loading.md
   understanding/indexing/indexing.md
   understanding/storing/storing.md
   understanding/querying/querying.md
   understanding/putting_it_all_together/putting_it_all_together.md
   understanding/tracing_and_debugging/tracing_and_debugging.md
   understanding/evaluating/evaluating.md

.. toctree::
   :maxdepth: 2
   :caption: Optimizing
   :hidden:

   optimizing/basic_strategies/basic_strategies.md
   optimizing/advanced_retrieval/advanced_retrieval.md
   optimizing/evaluation/evaluation.md
   optimizing/fine-tuning/fine-tuning.md

.. toctree::
   :maxdepth: 2
   :caption: Advanced techniques
   :hidden:

   advanced_techniques/development_pathway.md  
   advanced_techniques/production_rag.md 
   advanced_techniques/building_rag_from_scratch.md
   advanced_techniques/finetuning.md   

.. toctree::
   :maxdepth: 2
   :caption: Module Guides
   :hidden:

   module_guides/using_llms/using_llms.md
   module_guides/loading/loading.md
   module_guides/indexing/indexing.md
   module_guides/storing/storing.md
   module_guides/querying/querying.md
   module_guides/putting_it_all_together/putting_it_all_together.md
   module_guides/observability/observability.md
   module_guides/evaluating/evaluating.md
   module_guides/supporting_modules/supporting_modules.md

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
