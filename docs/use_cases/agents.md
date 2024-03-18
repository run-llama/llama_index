# Agents

An "agent" is an automated reasoning and decision engine. It takes in a user input/query and can make internal decisions for executing
that query in order to return the correct result. The key agent components can include, but are not limited to:

- Breaking down a complex question into smaller ones
- Choosing an external Tool to use + coming up with parameters for calling the Tool
- Planning out a set of tasks
- Storing previously completed tasks in a memory module

Research developments in LLMs (e.g. [ChatGPT Plugins](https://openai.com/blog/chatgpt-plugins)), LLM research ([ReAct](https://arxiv.org/abs/2210.03629), [Toolformer](https://arxiv.org/abs/2302.04761)) and LLM tooling ([LangChain](https://python.langchain.com/en/latest/modules/agents.html), [Semantic Kernel](https://github.com/microsoft/semantic-kernel)) have popularized the concept of agents.

## Agents + LlamaIndex

LlamaIndex provides some amazing tools to manage and interact with your data within your LLM application. And it is a core tool that you use while building an agent-based app.

- On one hand, many components within LlamaIndex are "agentic" - these make automated decisions to help a particular use case over your data. This ranges from simple reasoning (routing) to reasoning loops with memory (ReAct).
- On the other hand, LlamaIndex can be used as a core Tool within another agent framework.

## Resources

If you've built a RAG pipeline already and want to extend it with agentic behavior, check out the below resources

```{toctree}
---
maxdepth: 1
---
Agents (Putting your RAG Pipeline Together) </understanding/putting_it_all_together/agents.md>
Agentic Strategies (Optimizing your RAG Pipeline) </optimizing/agentic_strategies/agentic_strategies.md>
```

If you want to check out our standalone documentation hubs on agents and tools, check out the following module guides:

```{toctree}
---
maxdepth: 1
---
/module_guides/deploying/agents/root.md
/module_guides/deploying/agents/tools/root.md
```

## LlamaHub

We offer a collection of 40+ agent tools for use with your agent in [LlamaHub](https://llamahub.ai/) 🦙.
