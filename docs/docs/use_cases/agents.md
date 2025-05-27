# Agents

An "agent" is an automated reasoning and decision engine. It takes in a user input/query and can make internal decisions for executing
that query in order to return the correct result. The key agent components can include, but are not limited to:

- Breaking down a complex question into smaller ones
- Choosing an external Tool to use + coming up with parameters for calling the Tool
- Planning out a set of tasks
- Storing previously completed tasks in a memory module

LlamaIndex provides a comprehensive framework for building agentic systems with varying degrees of complexity:

- **If you want to build agents quickly**: Use our prebuilt [agent](../module_guides/deploying/agents/index.md) and [tool](../module_guides/deploying/agents/tools.md) architectures to rapidly setup agentic systems.
- **If you want full control over your agentic system**: Build and deploy custom agentic workflows from scratch using our [Workflows](../module_guides/workflow/index.md).


## Use Cases

The scope of possible use cases for agents is vast and ever-expanding. That said, here are some practical use cases that can deliver immediate value.

- **Agentic RAG**: Build a context-augmented research assistant over your data that not only answers simple questions, but complex research tasks. Our [getting started guide](../getting_started/starter_example.md) is a great place to start.

- **Report Generation**: Generate a multimodal report using a multi-agent researcher + writer workflow + LlamaParse. [Notebook](https://github.com/run-llama/llama_cloud_services/examples/parse/multimodal/multimodal_report_generation_agent.ipynb).

- **Customer Support**: Check out starter template for building a [multi-agent concierge with workflows](https://github.com/run-llama/multi-agent-concierge/).

Others:
- **Productivity Assistant**: Build an agent that can operate over common workflow tools like email, calendar. Check out our [GSuite agent tutorial](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-google/examples/advanced_tools_usage.ipynb).

- **Coding Assistant**: Build an agent that can operate over code. Check out our [code interpreter tutorial](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-code-interpreter/examples/code_interpreter.ipynb).


## Resources

**Prebuilt Agents and Tools**

The following component guides are the central hubs for getting started in building with agents:

- [Agents](../module_guides/deploying/agents/index.md)
- [Tools](../module_guides/deploying/agents/tools.md)


**Custom Agentic Workflows**

LlamaIndex Workflows allow you to build very custom, agentic workflows through a core event-driven orchestration foundation.

- [Workflows Tutorial Sequence](../understanding/workflows/index.md)
- [Workflows Component Guide](../module_guides/workflow/index.md)
- [Building a ReAct agent workflow](../examples/workflow/react_agent.ipynb)
- [Deploying Workflows](../module_guides/workflow/index.md#deploying-a-workflow)

**Building with Agentic Ingredients**

If you want to leverage core agentic ingredients in your workflow, LlamaIndex has robust abstractions for every agent sub-ingredient.

- **Query Planning**: [Routing](../module_guides/querying/router/index.md), [Sub-Questions](../examples/query_engine/sub_question_query_engine.ipynb), [Query Transformations](../optimizing/advanced_retrieval/query_transformations.md).
- **Function Calling and Tool Use**: Check out our [OpenAI](../examples/llm/openai.ipynb), [Mistral](../examples/llm/mistralai.ipynb) guides as examples.

## Ecosystem

- **Deploy Agents as Microservices**: Deploy your agentic workflows as microservices with [llama_deploy](../../module_guides/workflow/deployment.md) ([repo](https://github.com/run-llama/llama_deploy))
- **Community-Built Agents**: We offer a collection of 40+ agent tools for use with your agent in [LlamaHub](https://llamahub.ai/) 🦙.
