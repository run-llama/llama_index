# Agentic strategies

You can build agents on top of your existing LlamaIndex RAG pipeline to empower it with automated decision capabilities.
A lot of modules (routing, query transformations, and more) are already agentic in nature in that they use LLMs for decision making.

## Simpler Agentic Strategies

These include routing and query transformations.

```{toctree}
---
maxdepth: 1
---
/module_guides/querying/router/root.md
/optimizing/advanced_retrieval/query_transformations.md
Sub Question Query Engine (Intro) </examples/query_engine/sub_question_query_engine.ipynb>
```

## Data Agents

This guides below show you how to deploy a full agent loop, capable of chain-of-thought and query planning, on top of existing RAG query engines as tools for more advanced decision making.

Make sure to check out our [full module guide on Data Agents](/module_guides/deploying/agents/root.md), which highlight these use cases and much more.

Our [lower-level agent API](/module_guides/deploying/agents/agent_runner.md) shows you the internals of how an agent works (with step-wise execution).

Example guides below (using OpenAI function calling):

```{toctree}
---
maxdepth: 1
---
/examples/agent/openai_agent.ipynb
/examples/agent/openai_agent_with_query_engine.ipynb
/examples/agent/openai_agent_retrieval.ipynb
/examples/agent/openai_agent_query_cookbook.ipynb
/examples/agent/openai_agent_query_plan.ipynb
/examples/agent/openai_agent_context_retrieval.ipynb
```
