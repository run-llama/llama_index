# Deploying a Workflow

You can deploy a workflow as a multi-agent service with [llama_deploy](../../module_guides/llama_deploy) ([repo](https://github.com/run-llama/llama_deploy)). Each agent service is orchestrated via a control plane and communicates via a message queue. Deploy locally or on Kubernetes.

## Examples

To help you become more familiar with the workflow concept and its features, LlamaIndex documentation offers example
notebooks that you can run for hands-on learning:

- [Common Workflow Patterns](../../examples/workflow/workflows_cookbook.ipynb) walks you through common usage patterns
like looping and state management using simple workflows. It's usually a great place to start.
- [RAG + Reranking](../../examples/workflow/rag.ipynb) shows how to implement a real-world use case with a fairly
simple workflow that performs both ingestion and querying.
- [Citation Query Engine](../../examples/workflow/citation_query_engine.ipynb) similar to RAG + Reranking, the
notebook focuses on how to implement intermediate steps in between retrieval and generation. A good example of how to
use the [`Context`](#working-with-global-context-state) object in a workflow.
- [Corrective RAG](../../examples/workflow/corrective_rag_pack.ipynb) adds some more complexity on top of a RAG
workflow, showcasing how to query a web search engine after an evaluation step.
- [Utilizing Concurrency](../../examples/workflow/parallel_execution.ipynb) explains how to manage the parallel
execution of steps in a workflow, something that's important to know as your workflows grow in complexity.

RAG applications are easy to understand and offer a great opportunity to learn the basics of workflows. However, more complex agentic scenarios involving tool calling, memory, and routing are where workflows excel.

The examples below highlight some of these use-cases.

- [ReAct Agent](../../examples/workflow/react_agent.ipynb) is obviously the perfect example to show how to implement
tools in a workflow.
- [Function Calling Agent](../../examples/workflow/function_calling_agent.ipynb) is a great example of how to use the
LlamaIndex framework primitives in a workflow, keeping it small and tidy even in complex scenarios like function
calling.
- [CodeAct Agent](../../examples/agent/from_scratch_code_act_agent.ipynb) is a great example of how to create a CodeAct Agent from scratch.
- [Human In The Loop: Story Crafting](../../examples/workflow/human_in_the_loop_story_crafting.ipynb) is a powerful
example showing how workflow runs can be interactive and stateful. In this case, to collect input from a human.
- [Reliable Structured Generation](../../examples/workflow/reflection.ipynb) shows how to implement loops in a
workflow, in this case to improve structured output through reflection.
- [Query Planning with Workflows](../../examples/workflow/planning_workflow.ipynb) is an example of a workflow
that plans a query by breaking it down into smaller items, and executing those smaller items. It highlights how
to stream events from a workflow, execute steps in parallel, and looping until a condition is met.
- [Checkpointing Workflows](../../examples/workflow/checkpointing_workflows.ipynb) is a more exhaustive demonstration of how to make full use of `WorkflowCheckpointer` to checkpoint Workflow runs.

Last but not least, a few more advanced use cases that demonstrate how workflows can be extremely handy if you need
to quickly implement prototypes, for example from literature:

- [Advanced Text-to-SQL](../../examples/workflow/advanced_text_to_sql.ipynb)
- [JSON Query Engine](../../examples/workflow/JSONalyze_query_engine.ipynb)
- [Long RAG](../../examples/workflow/long_rag_pack.ipynb)
- [Multi-Step Query Engine](../../examples/workflow/multi_step_query_engine.ipynb)
- [Multi-Strategy Workflow](../../examples/workflow/multi_strategy_workflow.ipynb)
- [Router Query Engine](../../examples/workflow/router_query_engine.ipynb)
- [Self Discover Workflow](../../examples/workflow/self_discover_workflow.ipynb)
- [Sub-Question Query Engine](../../examples/workflow/sub_question_query_engine.ipynb)
