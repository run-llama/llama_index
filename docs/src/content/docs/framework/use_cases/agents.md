---
title: Agents
---

An "agent" is an automated reasoning and decision engine. It takes in a user input/query and can make internal decisions for executing
that query in order to return the correct result. The key agent components can include, but are not limited to:

- Breaking down a complex question into smaller ones
- Choosing an external Tool to use + coming up with parameters for calling the Tool
- Planning out a set of tasks
- Storing previously completed tasks in a memory module

LlamaIndex provides a comprehensive framework for building agentic systems with varying degrees of complexity:

- **If you want to build agents quickly**: Use our prebuilt [agent](/python/framework/module_guides/deploying/agents) and [tool](/python/framework/module_guides/deploying/agents/tools) architectures to rapidly setup agentic systems.
- **If you want full control over your agentic system**: Build and deploy custom agentic workflows from scratch using our [Workflows](/python/framework/module_guides/workflow).

## Use Cases

The scope of possible use cases for agents is vast and ever-expanding. That said, here are some practical use cases that can deliver immediate value.

- **Agentic RAG**: Build a context-augmented research assistant over your data that not only answers simple questions, but complex research tasks. Our [getting started guide](/python/framework/getting_started/starter_example) is a great place to start.

- **Report Generation**: Generate a multimodal report using a multi-agent researcher + writer workflow + LlamaParse. [Notebook](https://github.com/run-llama/llama_cloud_services/examples/parse/multimodal/multimodal_report_generation_agent.ipynb).

- **Customer Support**: Check out starter template for building a [multi-agent concierge with workflows](https://github.com/run-llama/multi-agent-concierge/).

Others:

- **Productivity Assistant**: Build an agent that can operate over common workflow tools like email, calendar. Check out our [GSuite agent tutorial](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-google/examples/advanced_tools_usage.ipynb).

- **Coding Assistant**: Build an agent that can operate over code. Check out our [code interpreter tutorial](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-code-interpreter/examples/code_interpreter.ipynb).

## Resources

**Prebuilt Agents and Tools**

The following component guides are the central hubs for getting started in building with agents:

- [Agents](/python/framework/module_guides/deploying/agents)
- [Tools](/python/framework/module_guides/deploying/agents/tools)

**Custom Agentic Workflows**

LlamaIndex Workflows allow you to build very custom, agentic workflows through a core event-driven orchestration foundation.

- [Workflows Documentation](/python/llamaagents/workflows)
- [Building a ReAct agent workflow](/python/examples/workflow/react_agent)
- [Deploying Workflows](/python/llamaagents/llamactl/getting-started/)

**Building with Agentic Ingredients**

If you want to leverage core agentic ingredients in your workflow, LlamaIndex has robust abstractions for every agent sub-ingredient.

- **Query Planning**: [Routing](/python/framework/module_guides/querying/router), [Sub-Questions](/python/examples/query_engine/sub_question_query_engine), [Query Transformations](/python/framework/optimizing/advanced_retrieval/query_transformations).
- **Function Calling and Tool Use**: Check out our [OpenAI](/python/examples/llm/openai), [Mistral](/python/examples/llm/mistralai) guides as examples.

## Ecosystem

- **Community-Built Agents**: We offer a collection of 40+ agent tools for use with your agent in [LlamaHub](https://llamahub.ai/) 🦙.
