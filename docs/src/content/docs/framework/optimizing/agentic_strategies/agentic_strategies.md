---
title: Agentic strategies
---

You can build agents on top of your existing LlamaIndex RAG workflow to empower it with automated decision capabilities.
A lot of modules (routing, query transformations, and more) are already agentic in nature in that they use LLMs for decision making.

## Simpler Agentic Strategies

These include routing and query transformations.

- [Routing](/python/framework/module_guides/querying/router)
- [Query Transformations](/python/framework/optimizing/advanced_retrieval/query_transformations)
- [Sub Question Query Engine (Intro)](/python/examples/query_engine/sub_question_query_engine)

## Data Agents

This guides below show you how to deploy a full agent loop, capable of chain-of-thought and query planning, on top of existing RAG query engines as tools for more advanced decision making.

Make sure to check out our [full module guide on Data Agents](/python/framework/module_guides/deploying/agents), which highlight these use cases and much more.

Our [lower-level agent API](/python/framework/module_guides/deploying/agents#manual-agents) shows you the internals of how an agent works (with step-wise execution).

Example guides below (using LLM-provider-specific function calling):

- [Basic Function Agent](/python/examples/workflow/function_calling_agent)
- [Function Agent with Query Engine Tools](/python/examples/agent/openai_agent_with_query_engine)
- [Function Agent Retrieval](/python/examples/agent/openai_agent_retrieval)
- [Function Agent Query Cookbook](/python/examples/agent/openai_agent_query_cookbook)
- [Function Agent w/ Context Retrieval](/python/examples/agent/openai_agent_context_retrieval)
