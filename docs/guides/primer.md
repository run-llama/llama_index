# A Primer to using GPT Index

At its core, GPT Index contains a toolkit of **index data structures** designed to easily connect LLM's with your external data.
GPT Index helps to provide the following advantages:
- Remove concerns over prompt size limitations.
- Abstract common usage patterns to reduce boilerplate code in your LLM app.
- Provide data connectors to your common data sources (Google Docs, Slack, etc.).
- Provide cost transparency + tools that reduce cost while increasing performance.


Each data structure offers distinct use cases and a variety of customizable parameters. These indices can then be 
*queried* in a general purpose manner, in order to achieve any task that you would typically achieve with an LLM:
- Question-Answering
- Summarization
- Text Generation (Stories, TODO's, emails, etc.)
- and more!

This primer is intended to help you get the most out of GPT Index. It gives a high-level overview of the following: 
1. The general usage pattern of GPT Index.
2. Mapping Use Cases to GPT Index data Structures
3. How Each Index Works


## 1. General Usage Pattern of GPT Index

The general usage pattern of GPT Index is as follows:
1. Load in documents (either manually, or through a data loader).
2. Index Construction.
3. [Optional, Advanced] Building indices on top of other indices
4. Query the index.

See our [Usage Pattern Guide](/guides/usage_pattern.md) for a guide
on the overall steps involved with using GPT Index.

If you are just starting out, take a look at the [Starter Example](/getting_started/starter_example.md) first.


## 2. Mapping Use Cases to GPT Index Data Structures

GPT Index data structures offer distinct use cases and advantages. For instance, the Vector Store-based indices e.g. `GPTSimpleVectorIndex` are a good general purpose tool for document retrieval. 
The list index `GPTListIndex` is a good tool for combining answers across documents/nodes. 
The tree index `GPTTreeIndex` and keyword indices can be used to "route" queries to the right subindices.

[A complete guide on GPT Index use cases](/guides/use_cases.md). 

This guide should paint a picture of how you can use GPT Index to solve your own data needs. 


## 3. How Each Index Works

We explain how each index works with diagrams.

[How Each Index Works](/guides/index_guide.md)

