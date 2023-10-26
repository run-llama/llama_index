# Agents

### "Agent-like" Components within LlamaIndex

LlamaIndex provides core modules capable of automated reasoning for different use cases over your data.

Some of these core modules are shown below along with example tutorials (not comprehensive, please click into the guides/how-tos for more details).

**SubQuestionQueryEngine for Multi-Document Analysis**

- [Sub Question Query Engine (Intro)](/examples/query_engine/sub_question_query_engine.ipynb)
- [10Q Analysis (Uber)](/examples/usecases/10q_sub_question.ipynb)
- [10K Analysis (Uber and Lyft)](/examples/usecases/10k_sub_question.ipynb)

**Query Transformations**

- [How-To](/optimizing/advanced_retrieval/query_transformations.md)
- [Multi-Step Query Decomposition](/examples/query_transformations/HyDEQueryTransformDemo.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/query_transformations/HyDEQueryTransformDemo.ipynb))

**Routing**

- [Usage](/module_guides/querying/router/root.md)
- [Router Query Engine Guide](/examples/query_engine/RouterQueryEngine.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/query_engine/RouterQueryEngine.ipynb))

**LLM Reranking**

- [Second Stage Processing How-To](/module_guides/querying/node_postprocessors/root.md)
- [LLM Reranking Guide (Great Gatsby)](/examples/node_postprocessor/LLMReranker-Gatsby.ipynb)

**Chat Engines**

- [Chat Engines How-To](/module_guides/deploying/chat_engines/root.md)

### Using LlamaIndex as as Tool within an Agent Framework

LlamaIndex can be used as as Tool within an agent framework - including LangChain, ChatGPT. These integrations are described below.

#### LangChain

We have deep integrations with LangChain.
LlamaIndex query engines can be easily packaged as Tools to be used within a LangChain agent, and LlamaIndex can also be used as a memory module / retriever. Check out our guides/tutorials below!

**Resources**

- [LangChain integration guide](/community/integrations/using_with_langchain.md)
- [Building a Chatbot Tutorial (LangChain + LlamaIndex)](/understanding/putting_it_all_together/chatbots/building_a_chatbot.md)
- [OnDemandLoaderTool Tutorial](/examples/tools/OnDemandLoaderTool.ipynb)

#### ChatGPT

LlamaIndex can be used as a ChatGPT retrieval plugin (we have a TODO to develop a more general plugin as well).

**Resources**

- [LlamaIndex ChatGPT Retrieval Plugin](https://github.com/openai/chatgpt-retrieval-plugin#llamaindex)

### Native OpenAIAgent

With the [new OpenAI API](https://openai.com/blog/function-calling-and-other-api-updates) that supports function calling, it’s never been easier to build your own agent!

Learn how to write your own OpenAI agent in **under 50 lines of code**, or directly use our super simple
`OpenAIAgent` implementation.
