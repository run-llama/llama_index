# LlamaIndex Agent Integration: Introspective Agent

## Introduction

This agent integration package includes three main agent classes:

1. `IntrospectiveAgentWorker`
2. `ToolInteractiveReflectionAgentWorker`
3. `SelfReflectionAgentWorker`

These classes are used together in order to build an "Introspective" Agent which
performs tasks while applying the reflection agentic pattern. In other words, an
introspective agent produces an initial response to a task and then performs
reflection and subsequently correction to produce an improved response to the task.

### The `IntrospectiveAgentWorker`

<p align="center">
  <img height="500" src="https://d3ddy8balm3goa.cloudfront.net/llamaindex/introspective_agents.excalidraw.svg" alt="cover">
</p>

This is the agent that is responsible for performing the task while utilizing the
reflection agentic pattern. It does so by merely delegating the work to two other
agents in a purely deterministic fashion.

Specifically, when given a task, this agent delegates the task to first a
`MainAgentWorker` that generates the initial response to the query. This initial
response is then passed to the `ReflectiveAgentWorker` to perform the reflection and
subsequent correction of the initial response. Optionally, the `MainAgentWorker`
can be skipped if none is provided. In this case, the users input query
will be assumed to contain the original response that needs to go thru
reflection and correction.

### The Reflection Agent Workers

These subclasses of the `BaseAgentWorker` are responsible for performing the
reflection and correction iterations of responses (starting with the initial
response from the `MainAgentWorker`). This package contains two reflection
agent workers: `ToolInteractiveReflectionAgentWorker` and `SelfReflectionAgentWorker`.

#### The `ToolInteractiveReflectionAgentWorker`

This agent worker implements the CRITIC reflection framework introduced
by Gou, Zhibin, et al. (2024) ICLR. (source: https://arxiv.org/pdf/2305.11738)

CRITIC stands for `Correcting with tool-interactive critiquing`. It works
by performing a reflection on a response to a task/query using external tools
(e.g., fact checking using a Google search tool) and subsequently using
the critique to generate a corrected response. It cycles thru tool-interactive
reflection and correction until a specific stopping criteria has been met
or a max number of iterations has been reached.

#### The `SelfReflectionAgentWorker`

This agent performs a reflection without any tools on a given response
and subsequently performs correction. Cycles of reflection and correction are
executed until a satisfactory correction has been generated or a max number of cycles
has been reached. To perform reflection, this agent utilizes a user-specified
LLM along with a PydanticProgram to generate a structured output that contains
an LLM generated reflection of the current response. After reflection, the
same user-specified LLM is used again but this time with another PydanticProgram
to generate a structured output that contains an LLM generated corrected
version of the current response against the priorly generated reflection.

### Usage

To build an introspective agent, we make use of the typical agent usage pattern,
where we construct an `IntrospectiveAgentWorker` and wrap it with an `AgentRunner`.
(Note this can be done convienently with the `.as_agent()` method of any `AgentWorker`
class.)

#### `IntrospectiveAgent` using `SelfReflectionAgentWorker`

```python
from llama_index.agent.introspective import IntrospectiveAgentWorker
from llama_index.agent.introspective import SelfReflectionAgentWorker
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgentWorker

verbose = True
self_reflection_agent_worker = SelfReflectionAgentWorker.from_defaults(
    llm=OpenAI("gpt-4-turbo-preview"),
    verbose=verbose,
)
main_agent_worker = OpenAIAgentWorker.from_tools(
    tools=[], llm=OpenAI("gpt-4-turbo-preview"), verbose=verbose
)

introspective_worker_agent = IntrospectiveAgentWorker.from_defaults(
    reflective_agent_worker=self_reflection_agent_worker,
    main_agent_worker=main_agent_worker,
    verbose=True,
)

introspective_agent = introspective_worker_agent.as_agent(verbose=verbose)
introspective_agent.chat("...")
```

#### `IntrospectiveAgent` using `ToolInteractiveReflectionAgentWorker`

Unlike with self reflection, here we need to define another agent worker,
namely the `CritiqueAgentWorker` that performs the reflection (or critique)
using a specified set of tools.

```python
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgentWorker
from llama_index.agent.introspective import (
    ToolInteractiveReflectionAgentWorker,
)
from llama_index.core.agent import FunctionCallingAgentWorker

verbose = True
critique_tools = []
critique_agent_worker = FunctionCallingAgentWorker.from_tools(
    tools=[critique_tools], llm=OpenAI("gpt-3.5-turbo"), verbose=verbose
)

correction_llm = OpenAI("gpt-4-turbo-preview")
tool_interactive_reflection_agent_worker = (
    ToolInteractiveReflectionAgentWorker.from_defaults(
        critique_agent_worker=critique_agent_worker,
        critique_template=(
            "..."
        ),  # template containing instructions for performing critique
        correction_llm=correction_llm,
        verbose=verbose,
    )
)


introspective_worker_agent = IntrospectiveAgentWorker.from_defaults(
    reflective_agent_worker=tool_interactive_reflection_agent_worker,
    main_agent_worker=None,  # if None, then its assumed user input is initial response
    verbose=verbose,
)
introspective_agent = introspective_worker_agent.as_agent(verbose=verbose)
introspective_agent.chat("...")
```
