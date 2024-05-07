# Agents

An "agent" is an automated reasoning and decision engine. It takes in a user input/query and can make internal decisions for executing
that query in order to return the correct result. The key agent components can include, but are not limited to:

- Breaking down a complex question into smaller ones
- Choosing an external Tool to use + coming up with parameters for calling the Tool
- Planning out a set of tasks
- Storing previously completed tasks in a memory module

LlamaIndex provides a comprehensive framework for building agents. This includes the following components:

- Using agents with tools at a high-level to build agentic RAG and workflow automation use cases
- Low-level components for building and debugging agents
- Core agent ingredients that can be used as standalone modules: query planning, tool use, and more.


## Use Cases

The scope of possible use cases for agents is vast and ever-expanding. That said, here are some practical use cases that can deliver immediate value.

- **Agentic RAG**: Build a context-augmented research assistant over your data that not only answers simple questions, but complex research tasks. Here are two resources ([resource 1](../understanding/putting_it_all_together/agents.md), [resource 2](../optimizing/agentic_strategies/agentic_strategies.md)) to help you get started.

- **SQL Agent**: A subset of the above is a "text-to-SQL assistant" that can interact with a structured database. Check out [this guide](https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/?h=sql+agent#setup-simple-retry-agent-pipeline-for-text-to-sql) to see how to build an agent from scratch.

- **Workflow Assistant**: Build an agent that can operate over common workflow tools like email, calendar. Check out our [GSuite agent tutorial](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-google/examples/advanced_tools_usage.ipynb).

- **Coding Assistant**: Build an agent that can operate over code. Check out our [code interpreter tutorial](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-code-interpreter/examples/code_interpreter.ipynb).


## Resources

**Using Agents with Tools**

The following component guides are the central hubs for getting started in building with agents:

- [Agents](../module_guides/deploying/agents/index.md)
- [Tools](../module_guides/deploying/agents/tools.md)


**Building Custom Agents**

If you're interested in building custom agents, check out the following resources.

- [Custom Agent](../examples/agent/custom_agent.ipynb)
- [Custom Agent with Query Pipelines](../examples/agent/agent_runner/query_pipeline_agent.ipynb)

**Building with Agentic Ingredients**

LlamaIndex has robust abstractions for every agent sub-ingredient.

- **Query Planning**: [Routing](../module_guides/querying/router/index.md), [Sub-Questions](../examples/query_engine/sub_question_query_engine.ipynb), [Query Transformations](../optimizing/advanced_retrieval/query_transformations.md).
- **Function Calling and Tool Use**: Check out our [OpenAI](../examples/llm/openai.ipynb), [Mistral](../examples/llm/mistralai.ipynb) guides as examples.
- **Memory**: [Example guide for adding memory to RAG](../examples/pipeline/query_pipeline_memory/).

## LlamaHub

We offer a collection of 40+ agent tools for use with your agent in [LlamaHub](https://llamahub.ai/) 🦙.
