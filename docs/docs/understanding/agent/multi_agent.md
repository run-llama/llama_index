# Multi-agent patterns in LlamaIndex

When more than one specialist is required to solve a task you have several options in LlamaIndex, each trading off convenience for flexibility.  This page walks through the three most common patterns, when to choose each one, and provides a minimal code sketch for every approach.

1. **AgentWorkflow (built-in)** – declare a set of agents and let `AgentWorkflow` manage the hand-offs.
2. **Orchestrator pattern (built-in)** – an "orchestrator" agent chooses which sub-agent to call next; those sub-agents are exposed to it as **tools**.
3. **Custom planner (DIY)** – you write the LLM prompt (often XML / JSON) that plans the sequence yourself and imperatively invoke the agents in code.

---

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

## Pattern 2 – Orchestrator agent (sub-agents as tools)

**When to use** – you want a single place that decides *every* step so you can inject custom logic, but you still prefer the declarative *agent as tool* experience over writing your own planner.

In this pattern you still build specialist agents (`ResearchAgent`, `WriteAgent`, `ReviewAgent`), **but** you do **not** ask them to hand off to one another.  Instead you expose each agent's `run` method as a tool and give those tools to a new top-level agent – the *Orchestrator*.

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

# assume research_agent / write_agent / review_agent defined as before

# Wrap each agent's `run` coroutine so it looks like a regular tool
research_tool = FunctionTool.from_defaults(
    fn=lambda msg: research_agent.run(user_msg=msg),
    name="call_research_agent",
    description="Use ResearchAgent to search the web and record notes.",
)


async def call_research_agent(ctx: Context, prompt: str) -> str:
    """Useful for recording research notes based on a specific prompt."""
    state = await ctx.get("state")
    if "research_notes" in state:
        return "Research notes already exist."

    result = await research_agent.run(
        user_msg=f"Write some notes about: {query}"
    )
    state["research_notes"] = str(result)
    await ctx.set("state", state)
    return str(result)


async def call_write_agent(ctx: Context) -> str:
    """Useful for writing a report based on the research notes."""
    state = await ctx.get("state")
    notes = state.get("research_notes")
    if not notes:
        return "No research notes to write from."

    result = await write_agent.run(
        user_msg=f"Write a report from the following notes: {notes}"
    )
    state["report_content"] = str(result)
    await ctx.set("state", state)
    return str(result)


async def call_review_agent(ctx: Context) -> str:
    """Useful for reviewing the report and providing feedback."""
    state = await ctx.get("state")
    report = state.get("report_content")
    if not report:
        return "No report content to review."

    result = await review_agent.run(
        user_msg=f"Review the following report: {report}"
    )
    state["review"] = result
    await ctx.set("state", state)
    return result


orchestrator = FunctionAgent(
    name="OrchestratorAgent",
    description="Decides which specialist to invoke next in order to fulfil the user's request.",
    system_prompt=(
        "You control three subordinate agents available as tools.  At each step, decide which one to call."
    ),
    llm=llm,
    tools=[research_tool, write_tool, review_tool],
)

response = await orchestrator.run(
    user_msg="Write me a report on the history of the web …"
)
print(response)
```

Because the orchestrator is just another `FunctionAgent` you get streaming, tool-calling, and state management for free – yet you keep full control over how agents are called and the overall control flow (tools always return back to the orchestrator).

---

## Pattern 3 – Custom planner (DIY prompting + parsing)

**When to use** – ultimate flexibility.  You need to impose a very specific plan format, integrate with external schedulers, or gather additional metadata that the previous patterns cannot provide out-of-the-box.

Here, the idea is that you write a prompt that instructs the LLM to output a structured plan (XML / JSON / YAML).  Your own Python code parses that plan and imperatively executes it.  The subordinate agents can be anything – `FunctionAgent`s, RAG pipelines, or other services.

Below is a *minimal* sketch that uses XML. Error-handling, retries and safety checks are left to you.

```python
import re
import xml.etree.ElementTree as ET
from llama_index.core.llms import ChatMessage, TextBlock

PLANNER_PROMPT = """
You are a planner. Given a user request and the current state, break the solution into ordered <step> blocks.  Each step must specify the agent to call and the message to send, e.g.
<plan>
  <step agent=\"ResearchAgent\">search for …</step>
  <step agent=\"WriteAgent\">draft a report …</step>
</plan>
Return ONLY valid XML.
"""

user_request = "Write me a report on the history of the web …"

state = {
    "research_notes": {},
    "report_content": "Not written yet.",
    "review": "Review required.",
}

chat_history = [
    ChatMessage(role="system", content=PLANNER_PROMPT),
    ChatMessage(
        role="user",
        blocks=[
            TextBlock(text=f"<state>\n{state}\n</state>\n"),
            TextBlock(text=user_request),
        ],
    ),
]

while True:
    planner_resp = await llm.achat(chat_history)
    chat_history.append(planner_resp.message)

    xml_str = re.search(
        r"<plan>(.*)</plan>", planner_resp.message.content, re.DOTALL
    ).group(1)

    if not xml_str:
        break

    root = ET.fromstring(xml_str)
    for step in root.findall("step"):
        agent_name = step.attrib["agent"]
        msg = step.text.strip()
        if agent_name == "ResearchAgent":
            result = await research_agent.run(user_msg=msg)
            state["research_notes"] = str(result)
        elif agent_name == "WriteAgent":
            result = await write_agent.run(user_msg=msg)
            state["report_content"] = str(result)
        elif agent_name == "ReviewAgent":
            result = await review_agent.run(user_msg=msg)
            state["review"] = str(result)

    # Update the chat history with the new state and prompt the LLM to plan again.
    chat_history.append(
        ChatMessage(
            role="user",
            blocks=[
                TextBlock(text=f"<state>\n{state}\n</state>\n"),
                TextBlock(text=user_request),
            ],
        )
    )

print(state)
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
