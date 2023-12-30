Welcome to LlamaIndex 🦙 !
##########################

LlamaIndex is a data framework for `LLM <https://en.wikipedia.org/wiki/Large_language_model>`_-based applications to ingest, structure, and access private or domain-specific data. It's available in Python (these docs) and `Typescript <https://ts.llamaindex.ai/>`_.

🚀 Why LlamaIndex?
******************

LLMs offer a natural language interface between humans and data. Widely available models come pre-trained on huge amounts of publicly available data like Wikipedia, mailing lists, textbooks, source code and more.

However, while LLMs are trained on a great deal of data, they are not trained on **your** data, which may be private or specific to the problem you're trying to solve. It's behind APIs, in SQL databases, or trapped in PDFs and slide decks.

You may choose to **fine-tune** a LLM with your data, but:

- Training a LLM is **expensive**.
- Due to the cost to train, it's **hard to update** a LLM with latest information.
- **Observability** is lacking. When you ask a LLM a question, it's not obvious how the LLM arrived at its answer.

LlamaIndex takes a different approach called `Retrieval-Augmented Generation (RAG) <./getting_started/concepts.html>`_. Instead of asking LLM to generate an answer immediately, LlamaIndex:

1. retrieves information from your data sources first,
2. adds it to your question as context, and
3. asks the LLM to answer based on the enriched prompt.

RAG overcomes all three weaknesses of the fine-tuning approach:

- There's no training involved, so it's **cheap**.
- Data is fetched only when you ask for them, so it's **always up to date**.
- LlamaIndex can show you the retrieved documents, so it's **more trustworthy**.

LlamaIndex imposes no restriction on how you use LLMs. You can still use LLMs as auto-complete, chatbots, semi-autonomous agents, and more (see Use Cases on the left). It only makes LLMs more relevant to you.

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
*******************************************

LlamaIndex provides tools for beginners, advanced users, and everyone in between.

Our high-level API allows beginner users to use LlamaIndex to ingest and query their data in 5 lines of code.

For more complex applications, our lower-level APIs allow advanced users to customize and extend any module—data connectors, indices, retrievers, query engines, reranking modules—to fit their needs.

Getting Started
****************

To install the library:

``pip install llama-index``

We recommend starting at `how to read these docs <./getting_started/reading.html>`_, which will point you to the right place based on your experience level.

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
   getting_started/reading.md
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
   use_cases/extraction.md
   use_cases/multimodal.md

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
   optimizing/agentic_strategies/agentic_strategies.md
   optimizing/evaluation/evaluation.md
   optimizing/fine-tuning/fine-tuning.md
   optimizing/production_rag.md
   optimizing/building_rag_from_scratch.md
.. toctree::
   :maxdepth: 2
   :caption: Module Guides
   :hidden:

   module_guides/models/models.md
   module_guides/models/prompts.md
   module_guides/loading/loading.md
   module_guides/indexing/indexing.md
   module_guides/storing/storing.md
   module_guides/querying/querying.md
   module_guides/observability/observability.md
   module_guides/evaluating/root.md
   module_guides/supporting_modules/supporting_modules.md


.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api_reference/index.rst

.. toctree::
   :maxdepth: 2
   :caption: Community
   :hidden:

   community/integrations.md
   community/frequently_asked_questions.md
   community/full_stack_projects.md

.. toctree::
   :maxdepth: 2
   :caption: Contributing
   :hidden:

   contributing/contributing.rst
   contributing/documentation.rst

.. toctree::
   :maxdepth: 2
   :caption: Changes
   :hidden:

   changes/changelog.rst
   changes/deprecated_terms.md
