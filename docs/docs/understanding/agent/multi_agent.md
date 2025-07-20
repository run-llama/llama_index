# Multi-agent patterns in LlamaIndex

When more than one specialist is required to solve a task you have several options in LlamaIndex, each trading off convenience for flexibility.  This page walks through the three most common patterns, when to choose each one, and provides a minimal code sketch for every approach.

1. **AgentWorkflow (built-in)** – declare a set of agents and let `AgentWorkflow` manage the hand-offs. [Section](#pattern-1--agentworkflow-ie-linear-swarm-pattern) [Full Notebook](../../examples/agent/agent_workflow_multi.ipynb)
2. **Orchestrator pattern (built-in)** – an "orchestrator" agent chooses which sub-agent to call next; those sub-agents are exposed to it as **tools**. [Section](#pattern-2--orchestrator-agent-sub-agents-as-tools) [Full Notebook](../../examples/agent/agents_as_tools.ipynb)
3. **Custom planner (DIY)** – you write the LLM prompt (often XML / JSON) that plans the sequence yourself and imperatively invoke the agents in code. [Section](#pattern-3--custom-planner-diy-prompting--parsing) [Full Notebook](../../examples/agent/custom_multi_agent.ipynb)

---

<div id="pattern-1--agentworkflow-ie-linear-swarm-pattern"></div>
## Pattern 1 – AgentWorkflow (i.e. linear "swarm" pattern)

**When to use** – you want multi-agent behaviour out-of-the-box with almost no extra code, and you are happy with the default hand-off heuristics that ship with `AgentWorkflow`.

`AgentWorkflow` is itself a [Workflow](../workflows/index.md) pre-configured to understand agents, state and tool-calling.  You supply an *array* of one or more agents, tell it which one should start, and it will:

1. Give the *root* agent the user message.
2. Execute whatever tools that agent selects.
3. Allow the agent to "handoff" control to another agent when it decides.
4. Repeat until an agent returns a final answer.

**NOTE:** At any point, the current active agent can choose to return control back to the user.

Below is the condensed version of the [multi-agent report generation example](../../examples/agent/agent_workflow_multi.ipynb).  Three agents collaborate to research, write and review a report.  (`…` indicates code omitted for brevity.)

```python
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent

# --- create our specialist agents ------------------------------------------------
research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Search the web and record notes.",
    system_prompt="You are a researcher… hand off to WriteAgent when ready.",
    llm=llm,
    tools=[search_web, record_notes],
    can_handoff_to=["WriteAgent"],
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Writes a markdown report from the notes.",
    system_prompt="You are a writer… ask ReviewAgent for feedback when done.",
    llm=llm,
    tools=[write_report],
    can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Reviews a report and gives feedback.",
    system_prompt="You are a reviewer…",  # etc.
    llm=llm,
    tools=[review_report],
    can_handoff_to=["WriteAgent"],
)

# --- wire them together ----------------------------------------------------------
agent_workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent.name,
    initial_state={
        "research_notes": {},
        "report_content": "Not written yet.",
        "review": "Review required.",
    },
)

resp = await agent_workflow.run(
    user_msg="Write me a report on the history of the web …"
)
print(resp)
```

`AgentWorkflow` does all the orchestration, streaming events as it goes so you can keep users informed of progress.

---

<div id="pattern-2--orchestrator-agent-sub-agents-as-tools"></div>
## Pattern 2 – Orchestrator agent (sub-agents as tools)

**When to use** – you want a single place that decides *every* step so you can inject custom logic, but you still prefer the declarative *agent as tool* experience over writing your own planner.

In this pattern you still build specialist agents (`ResearchAgent`, `WriteAgent`, `ReviewAgent`), **but** you do **not** ask them to hand off to one another.  Instead you expose each agent's `run` method as a tool and give those tools to a new top-level agent – the *Orchestrator*.

You can see the full example in the [agents_as_tools notebook](../../examples/agent/agents_as_tools.ipynb).

```python
import re
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context

# assume research_agent / write_agent / review_agent defined as before
# except we really only need the `search_web` tool at a minimum


async def call_research_agent(ctx: Context, prompt: str) -> str:
    """Useful for recording research notes based on a specific prompt."""
    result = await research_agent.run(
        user_msg=f"Write some notes about the following: {prompt}"
    )

    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["research_notes"].append(str(result))

    return str(result)


async def call_write_agent(ctx: Context) -> str:
    """Useful for writing a report based on the research notes or revising the report based on feedback."""
    async with ctx.store.edit_state() as ctx_state:
        notes = ctx_state["state"].get("research_notes", None)
        if not notes:
            return "No research notes to write from."

        user_msg = f"Write a markdown report from the following notes. Be sure to output the report in the following format: <report>...</report>:\n\n"

        # Add the feedback to the user message if it exists
        feedback = ctx_state["state"].get("review", None)
        if feedback:
            user_msg += f"<feedback>{feedback}</feedback>\n\n"

        # Add the research notes to the user message
        notes = "\n\n".join(notes)
        user_msg += f"<research_notes>{notes}</research_notes>\n\n"

        # Run the write agent
        result = await write_agent.run(user_msg=user_msg)
        report = re.search(
            r"<report>(.*)</report>", str(result), re.DOTALL
        ).group(1)
        ctx_state["state"]["report_content"] = str(report)

    return str(report)


async def call_review_agent(ctx: Context) -> str:
    """Useful for reviewing the report and providing feedback."""
    async with ctx.store.edit_state() as ctx_state:
        report = ctx_state["state"].get("report_content", None)
        if not report:
            return "No report content to review."

        result = await review_agent.run(
            user_msg=f"Review the following report: {report}"
        )
        ctx_state["state"]["review"] = result

    return result


orchestrator = FunctionAgent(
    system_prompt=(
        "You are an expert in the field of report writing. "
        "You are given a user request and a list of tools that can help with the request. "
        "You are to orchestrate the tools to research, write, and review a report on the given topic. "
        "Once the review is positive, you should notify the user that the report is ready to be accessed."
    ),
    llm=orchestrator_llm,
    tools=[
        call_research_agent,
        call_write_agent,
        call_review_agent,
    ],
    initial_state={
        "research_notes": [],
        "report_content": None,
        "review": None,
    },
)

response = await orchestrator.run(
    user_msg="Write me a report on the history of the web …"
)
print(response)
```

Because the orchestrator is just another `FunctionAgent` you get streaming, tool-calling, and state management for free – yet you keep full control over how agents are called and the overall control flow (tools always return back to the orchestrator).

---

<div id="pattern-3--custom-planner-diy-prompting--parsing"></div>
## Pattern 3 – Custom planner (DIY prompting + parsing)

**When to use** – ultimate flexibility.  You need to impose a very specific plan format, integrate with external schedulers, or gather additional metadata that the previous patterns cannot provide out-of-the-box.

Here, the idea is that you write a prompt that instructs the LLM to output a structured plan (XML / JSON / YAML).  Your own Python code parses that plan and imperatively executes it.  The subordinate agents can be anything – `FunctionAgent`s, RAG pipelines, or other services.

Below is a *minimal* sketch of a workflow that can plan, execute a plan, and see if any further steps are needed. You can see the full example in the [custom_multi_agent notebook](../../examples/agent/custom_multi_agent.ipynb).

```python
import re
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
from typing import Any, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

# Assume we created helper functions to call the agents

PLANNER_PROMPT = """You are a planner chatbot.

Given a user request and the current state, break the solution into ordered <step> blocks.  Each step must specify the agent to call and the message to send, e.g.
<plan>
  <step agent=\"ResearchAgent\">search for …</step>
  <step agent=\"WriteAgent\">draft a report …</step>
  ...
</plan>

<state>
{state}
</state>

<available_agents>
{available_agents}
</available_agents>

The general flow should be:
- Record research notes
- Write a report
- Review the report
- Write the report again if the review is not positive enough

If the user request does not require any steps, you can skip the <plan> block and respond directly.
"""


class InputEvent(StartEvent):
    user_msg: Optional[str] = Field(default=None)
    chat_history: list[ChatMessage]
    state: Optional[dict[str, Any]] = Field(default=None)


class OutputEvent(StopEvent):
    response: str
    chat_history: list[ChatMessage]
    state: dict[str, Any]


class StreamEvent(Event):
    delta: str


class PlanEvent(Event):
    step_info: str


# Modelling the plan
class PlanStep(BaseModel):
    agent_name: str
    agent_input: str


class Plan(BaseModel):
    steps: list[PlanStep]


class ExecuteEvent(Event):
    plan: Plan
    chat_history: list[ChatMessage]


class PlannerWorkflow(Workflow):
    llm: OpenAI = OpenAI(
        model="o3-mini",
        api_key="sk-proj-...",
    )
    agents: dict[str, FunctionAgent] = {
        "ResearchAgent": research_agent,
        "WriteAgent": write_agent,
        "ReviewAgent": review_agent,
    }

    @step
    async def plan(
        self, ctx: Context, ev: InputEvent
    ) -> ExecuteEvent | OutputEvent:
        # Set initial state if it exists
        if ev.state:
            await ctx.store.set("state", ev.state)

        chat_history = ev.chat_history

        if ev.user_msg:
            user_msg = ChatMessage(
                role="user",
                content=ev.user_msg,
            )
            chat_history.append(user_msg)

        # Inject the system prompt with state and available agents
        state = await ctx.store.get("state")
        available_agents_str = "\n".join(
            [
                f'<agent name="{agent.name}">{agent.description}</agent>'
                for agent in self.agents.values()
            ]
        )
        system_prompt = ChatMessage(
            role="system",
            content=PLANNER_PROMPT.format(
                state=str(state),
                available_agents=available_agents_str,
            ),
        )

        # Stream the response from the llm
        response = await self.llm.astream_chat(
            messages=[system_prompt] + chat_history,
        )
        full_response = ""
        async for chunk in response:
            full_response += chunk.delta or ""
            if chunk.delta:
                ctx.write_event_to_stream(
                    StreamEvent(delta=chunk.delta),
                )

        # Parse the response into a plan and decide whether to execute or output
        xml_match = re.search(r"(<plan>.*</plan>)", full_response, re.DOTALL)

        if not xml_match:
            chat_history.append(
                ChatMessage(
                    role="assistant",
                    content=full_response,
                )
            )
            return OutputEvent(
                response=full_response,
                chat_history=chat_history,
                state=state,
            )
        else:
            xml_str = xml_match.group(1)
            root = ET.fromstring(xml_str)
            plan = Plan(steps=[])
            for step in root.findall("step"):
                plan.steps.append(
                    PlanStep(
                        agent_name=step.attrib["agent"],
                        agent_input=step.text.strip() if step.text else "",
                    )
                )

            return ExecuteEvent(plan=plan, chat_history=chat_history)

    @step
    async def execute(self, ctx: Context, ev: ExecuteEvent) -> InputEvent:
        chat_history = ev.chat_history
        plan = ev.plan

        for step in plan.steps:
            agent = self.agents[step.agent_name]
            agent_input = step.agent_input
            ctx.write_event_to_stream(
                PlanEvent(
                    step_info=f'<step agent="{step.agent_name}">{step.agent_input}</step>'
                ),
            )

            if step.agent_name == "ResearchAgent":
                await call_research_agent(ctx, agent_input)
            elif step.agent_name == "WriteAgent":
                # Note: we aren't passing the input from the plan since
                # we're using the state to drive the write agent
                await call_write_agent(ctx)
            elif step.agent_name == "ReviewAgent":
                await call_review_agent(ctx)

        state = await ctx.store.get("state")
        chat_history.append(
            ChatMessage(
                role="user",
                content=f"I've completed the previous steps, here's the updated state:\n\n<state>\n{state}\n</state>\n\nDo you need to continue and plan more steps?, If not, write a final response.",
            )
        )

        return InputEvent(
            chat_history=chat_history,
        )
```

This approach means *you* own the orchestration loop, so you can insert whatever custom logic, caching or human-in-the-loop checks you require.

---

## Choosing a pattern

| Pattern | Lines of code | Flexibility | Built-in streaming / events |
|---------|--------------|-------------|-----------------------------|
| AgentWorkflow | ⭐ – fewest | ★★ | Yes |
| Orchestrator agent | ⭐⭐ | ★★★ | Yes (via orchestrator) |
| Custom planner | ⭐⭐⭐ | ★★★★★ | Yes (via sub-agents). Top-level is up to you. |

If you are prototyping quickly, start with `AgentWorkflow`.  Move to an *Orchestrator agent* when you need more control over the sequence.  Reach for a *Custom planner* only when the first two patterns cannot express the flow you need.

Next you will learn how to use [structured output in single and multi-agent workflows](./structured_output.md)
