# Agents

Putting together an agent in LlamaIndex can be done by defining a set of tools and providing them to our ReActAgent implementation. We're using it here with OpenAI, but it can be used with any sufficiently capable LLM:

```python
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent


# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

# initialize llm
llm = OpenAI(model="gpt-3.5-turbo-0613")

# initialize ReAct agent
agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)
```

These tools can be Python functions as shown above, or they can be LlamaIndex query engines:

```python
from llama_index.core.tools import QueryEngineTool

query_engine_tools = [
    QueryEngineTool(
        query_engine=sql_agent,
        metadata=ToolMetadata(
            name="sql_agent", description="Agent that can execute SQL queries."
        ),
    ),
]

agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
```

You can learn more in our [Agent Module Guide](../../module_guides/deploying/agents/index.md).

## Native OpenAIAgent

We have an `OpenAIAgent` implementation built on the [OpenAI API for function calling](https://openai.com/blog/function-calling-and-other-api-updates) that allows you to rapidly build agents:

- [OpenAIAgent](../../examples/agent/openai_agent.ipynb)
- [OpenAIAgent with Query Engine Tools](../../examples/agent/openai_agent_with_query_engine.ipynb)
- [OpenAIAgent Query Planning](../../examples/agent/openai_agent_query_plan.ipynb)
- [OpenAI Assistant](../../examples/agent/openai_assistant_agent.ipynb)
- [OpenAI Assistant Cookbook](../../examples/agent/openai_assistant_query_cookbook.ipynb)
- [Forced Function Calling](../../examples/agent/openai_forced_function_call.ipynb)
- [Parallel Function Calling](../../examples/agent/openai_agent_parallel_function_calling.ipynb)
- [Context Retrieval](../../examples/agent/openai_agent_context_retrieval.ipynb)

## Agentic Components within LlamaIndex

LlamaIndex provides core modules capable of automated reasoning for different use cases over your data which makes them essentially Agents. Some of these core modules are shown below along with example tutorials.

**SubQuestionQueryEngine for Multi Document Analysis**

- [Sub Question Query Engine (Intro)](../../examples/query_engine/sub_question_query_engine.ipynb)
- [10Q Analysis (Uber)](../../examples/usecases/10q_sub_question.ipynb)
- [10K Analysis (Uber and Lyft)](../../examples/usecases/10k_sub_question.ipynb)

**Query Transformations**

- [How-To](../../optimizing/advanced_retrieval/query_transformations.md)
- [Multi-Step Query Decomposition](../../examples/query_transformations/HyDEQueryTransformDemo.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/docs/examples/query_transformations/HyDEQueryTransformDemo.ipynb))

**Routing**

- [Usage](../../module_guides/querying/router/index.md)
- [Router Query Engine Guide](../../examples/query_engine/RouterQueryEngine.ipynb) ([Notebook](https://github.com/jerryjliu/llama_index/blob/main/docs../../examples/query_engine/RouterQueryEngine.ipynb))

**LLM Reranking**

- [Second Stage Processing How-To](../../module_guides/querying/node_postprocessors/index.md)
- [LLM Reranking Guide (Great Gatsby)](../../examples/node_postprocessor/LLMReranker-Gatsby.ipynb)

**Chat Engines**

- [Chat Engines How-To](../../module_guides/deploying/chat_engines/index.md)

## Using LlamaIndex as a Tool within an Agent Framework

LlamaIndex can be used as a Tool within an agent framework - including LangChain, ChatGPT. These integrations are described below.

### LangChain

We have deep integrations with LangChain.
LlamaIndex query engines can be easily packaged as Tools to be used within a LangChain agent, and LlamaIndex can also be used as a memory module / retriever. Check out our guides/tutorials below!

**Resources**

- [Building a Chatbot Tutorial](chatbots/building_a_chatbot.md)
- [OnDemandLoaderTool Tutorial](../../examples/tools/OnDemandLoaderTool.ipynb)

### ChatGPT

LlamaIndex can be used as a ChatGPT retrieval plugin (we have a TODO to develop a more general plugin as well).

**Resources**

- [LlamaIndex ChatGPT Retrieval Plugin](https://github.com/openai/chatgpt-retrieval-plugin#llamaindex)
