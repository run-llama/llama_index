# Agents

Putting together an agent in LlamaIndex can be done by defining a set of tools and providing them to our ReActAgent or FunctionAgent implementation. We're using it here with OpenAI, but it can be used with any sufficiently capable LLM.

In general, FunctionAgent should be preferred for LLMs that have built-in function calling/tools in their API, like Openai, Anthropic, Gemini, etc.

```python
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent


# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


# initialize llm
llm = OpenAI(model="gpt-4o")

# initialize agent
agent = FunctionAgent(
    tools=[multiply],
    system_prompt="You are an agent that can invoke a tool for multiplication when assisting a user.",
)
```

These tools can be Python functions as shown above, or they can be LlamaIndex query engines:

```python
from llama_index.core.tools import QueryEngineTool

query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=sql_agent,
        name="sql_agent",
        description="Agent that can execute SQL queries.",
    ),
]

agent = FunctionAgent(
    tools=query_engine_tools,
    system_prompt="You are an agent that can invoke an agent for text-to-SQL execution.",
)
```

You can learn more in our [Agent Module Guide](../../module_guides/deploying/agents/index.md) or in our [end-to-end agent tutorial](../agent/index.md).

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
