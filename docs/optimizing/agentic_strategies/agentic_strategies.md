# Agentic strategies

You can build agents on top of your existing LlamaIndex RAG pipeline to empower it with automated decision capabilities.
A lot of modules (routing, query transformations, and more) are already agentic in nature in that they use LLMs for decision making.

This section shows you how to deploy a full agent loop, capable of chain-of-thought and query planning, on top of existing RAG query engines as tools for more advanced decision making.

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
