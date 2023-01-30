.. GPT Index documentation master file, created by
   sphinx-quickstart on Sun Dec 11 14:30:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GPT Index!
=====================================

GPT Index is a project consisting of a set of data structures designed to make it easier to 
use large external knowledge bases with LLMs.

- Github: https://github.com/jerryjliu/gpt_index
- PyPi: https://pypi.org/project/gpt-index/
- Twitter: https://twitter.com/gpt_index
- Discord https://discord.gg/dGcwcsnxhU


🚀 Overview
-----------

Context
^^^^^^^
- LLMs are a phenomenonal piece of technology for knowledge generation and reasoning.
- A big limitation of LLMs is context size (e.g. Davinci's limit is 4096 tokens. Large, but not infinite).
- The ability to feed "knowledge" to LLMs is restricted to this limited prompt size and model weights.

Proposed Solution
^^^^^^^^^^^^^^^^^
That's where the **GPT Index** comes in. GPT Index is a simple, flexible interface between your external data and LLMs. It resolves the following pain points:

- Provides simple data structures to resolve prompt size limitations.
- Offers data connectors to your external data sources.
- Offers you a comprehensive toolset trading off cost and performance.

At the core of GPT Index is a **data structure**. Instead of relying on world knowledge encoded in the model weights, a GPT Index data structure does the following:

- Uses a pre-trained LLM primarily for *reasoning*/*summarization* instead of prior knowledge.
- Takes as input a large corpus of text data and build a structured index over it (using an LLM or heuristics).
- Allow users to *query* the index by passing in an LLM prompt, and obtaining a response.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation.md
   getting_started/starter_example.md


.. toctree::
   :maxdepth: 1
   :caption: Guides

   guides/primer.md 
   guides/usage_pattern.md
   guides/use_cases.md
   guides/index_guide.md


.. toctree::
   :maxdepth: 1
   :caption: Technical How To

   how_to/data_connectors.md
   how_to/composability.md
   how_to/update.md
   how_to/cost_analysis.md
   how_to/embeddings.md
   how_to/vector_stores.md
   how_to/custom_prompts.md
   how_to/custom_llms.md
   how_to/using_with_langchain.md


.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/indices.rst
   reference/query.rst
   reference/composability.rst
   reference/readers.rst
   reference/prompts.rst
   reference/llm_predictor.rst
   reference/prompt_helper.rst
   reference/embeddings.rst
   reference/struct_store.rst
   reference/response.rst
   reference/example_notebooks.rst


.. toctree::
   :maxdepth: 1
   :caption: Gallery

   gallery/app_showcase.md
