.. GPT Index documentation master file, created by
   sphinx-quickstart on Sun Dec 11 14:30:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GPT Index!
=====================================

GPT Index is a project consisting of a set of data structures that are created using GPT-3 and can be traversed using GPT-3 in order to answer queries.


🚀 Overview
-----------

Context
^^^^^^^
- LLM's are a phenomenonal piece of technology for knowledge generation and reasoning.
- A big limitation of LLM's is context size (e.g. Davinci's limit is 4096 tokens. Large, but not infinite).
- The ability to feed "knowledge" to LLM's is restricted to this limited prompt size and model weights.
- **Thought**: What if LLM's can have access to potentially a much larger database of knowledge without retraining/finetuning? 

Proposed Solution
^^^^^^^^^^^^^^^^^
That's where the **GPT Index** comes in. GPT Index is a simple, flexible interface between your external data and LLM's. It resolves the following pain points:
- Provides simple data structures to resolve prompt size limitations.
- Offers data connectors to your external data sources.
- Offers you a comprehensive toolset trading off cost and performance.

At the core of GPT Index is a **data structure**. Instead of relying on world knowledge encoded in the model weights, a GPT Index data structure does the following:
- Uses a pre-trained LLM primarily for *reasoning*/*summarization* instead of prior knowledge.
- Takes as input a large corpus of text data and build a structured index over it (using an LLM or heuristics).
- Allow users to _query_ the index in order to synthesize an answer to the question - this requires both _traversal_ of the index as well as a synthesis of the answer.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation.md
   getting_started/starter_example.md


.. toctree::
   :maxdepth: 1
   :caption: How To Guides

   how_to/data_connectors.md
   how_to/composability.md
   how_to/insert.md
   how_to/cost_analysis.md


.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/indices.rst
   reference/readers.rst

