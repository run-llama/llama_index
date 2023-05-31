# Agents

## Context
An "agent" is an automated reasoning and decision engine. It takes in a user input/query and can make internal decisions for executing
that query in order to return the correct result. The key agent components can include, but are not limited to:
- Breaking down a complex question into smaller ones
- Choosing an external Tool to use + coming up with parameters for calling the Tool
- Planning out a set of tasks
- Storing previously completed tasks in a memory module

Research developments in LLMs (e.g. [ChatGPT Plugins](https://openai.com/blog/chatgpt-plugins)), LLM research ([ReAct](https://arxiv.org/abs/2210.03629), [Toolformer](https://arxiv.org/abs/2302.04761)) and LLM tooling ([LangChain](https://python.langchain.com/en/latest/modules/agents.html), [Semantic Kernel](https://github.com/microsoft/semantic-kernel)) have popularized the concept of agents.

## Agents + LlamaIndex

LlamaIndex provides some amazing tools to manage and interact with your data within your LLM application. And it can be a core tool that you use while building an agent-based app.
- On one hand, some components within LlamaIndex are "agent-like" - these make automated decisions to help a particular use case over your data.
- On the other hand, LlamaIndex can be used as a core Tool within another agent framework.

In general, LlamaIndex components offer more explicit, constrained behavior for more specific use cases. Agent frameworks such as ReAct (implemented in LangChain) offer agents that are more unconstrained + 
capable of general reasoning. 

There are tradeoffs for using both - less-capable LLMs typically do better with more constraints. Take a look at [our blog post on this](https://medium.com/llamaindex-blog/dumber-llm-agents-need-more-constraints-and-better-tools-17a524c59e12) for 
a more information + a detailed analysis.


### "Agent-like" Components within LlamaIndex 

LlamaIndex provides core modules capable of automated reasoning for different use cases over your data. Please check out our [use cases doc](/use_cases/queries.md) for more details on high-level use cases that LlamaIndex can help fulfill.

Some of these core modules are shown below along with example tutorials (not comprehensive, please click into the guides/how-tos for more details).

**SubQuestionQueryEngine for Multi-Document Analysis**
- [Usage](queries.md#multi-document-queries)
- [Sub Question Query Engine (Intro)](/examples/query_engine/sub_question_query_engine.ipynb)
- [10Q Analysis (Uber)](/examples/usecases/10q_sub_question.ipynb)
- [10K Analysis (Uber and Lyft)](/examples/usecases/10k_sub_question.ipynb)


**Query Transformations**
- [How-To](/how_to/query/query_transformations.md)
- [Multi-Step Query Decomposition](/examples/query_transformations/HyDEQueryTransformDemo.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/query_transformations/HyDEQueryTransformDemo.ipynb))

**Routing**
- [Usage](queries.md#routing-over-heterogeneous-data)
- [Router Query Engine Guide](/examples/query_engine/RouterQueryEngine.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/query_engine/RouterQueryEngine.ipynb))

**LLM Reranking**
- [Second Stage Processing How-To](/how_to/query/second_stage.md)
- [LLM Reranking Guide (Great Gatsby)](/examples/node_postprocessor/LLMReranker-Gatsby.ipynb)

**Chat Engines**
- [Chat Engines How-To](/how_to/query/chat_engines.md)


### Using LlamaIndex as as Tool within an Agent Framework

LlamaIndex can be used as as Tool within an agent framework - including LangChain, ChatGPT. These integrations are described below.

#### LangChain

We have deep integrations with LangChain. 
LlamaIndex query engines can be easily packaged as Tools to be used within a LangChain agent, and LlamaIndex can also be used as a memory module / retriever. Check out our guides/tutorials below!

**Resources**
- [LangChain integration guide](/how_to/integrations/using_with_langchain.md)
- [Building a Chatbot Tutorial (LangChain + LlamaIndex)](/guides/tutorials/building_a_chatbot.md)

#### ChatGPT

LlamaIndex can be used as a ChatGPT retrieval plugin (we have a TODO to develop a more general plugin as well).

**Resources**
- [LlamaIndex ChatGPT Retrieval Plugin](https://github.com/openai/chatgpt-retrieval-plugin#llamaindex)


