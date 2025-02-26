# Multi-agent systems with AgentWorkflow

So far you've been using `AgentWorkflow` to create single agents. But `AgentWorkflow` is also designed to support multi-agent systems, where multiple agents collaborate to complete your task, handing off control to each other as needed.

In this example, our system will have three agents:

* A `ResearchAgent` that will search the web for information on the given topic.
* A `WriteAgent` that will write the report using the information found by the ResearchAgent.
* A `ReviewAgent` that will review the report and provide feedback.

We will use  `AgentWorkflow` to create a multi-agent system that will execute these agents in order.

There are a lot of ways we could go about building a system to perform this task. In this example, we will use a few tools to help with the research and writing processes.

* A `web_search` tool to search the web for information on the given topic (we'll use Tavily, as we did in previous examples)
* A `record_notes` tool which will save research found on the web to the state so that the other tools can use it (see [state management](./state.md) to remind yourself how this works)
* A `write_report` tool to write the report using the information found by the `ResearchAgent`
* A `review_report` tool to review the report and provide feedback.

Utilizing the Context class, we can pass state between agents, and each agent will have access to the current state of the system.

We'll define our `web_search` tool simply by using the one we get from the `TavilyToolSpec`:

```python
tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))
search_web = tavily_tool.to_tool_list()[0]
```

Our `record_notes` tool will access the current state, add the notes to the state, and then return a message indicating that the notes have been recorded.

```python
async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
    """Useful for recording notes on a given topic."""
    current_state = await ctx.get("state")
    if "research_notes" not in current_state:
        current_state["research_notes"] = {}
    current_state["research_notes"][notes_title] = notes
    await ctx.set("state", current_state)
    return "Notes recorded."
```

`write_report` and `review_report` will similarly be tools that access the state:

```python
async def write_report(ctx: Context, report_content: str) -> str:
    """Useful for writing a report on a given topic."""
    current_state = await ctx.get("state")
    current_state["report_content"] = report_content
    await ctx.set("state", current_state)
    return "Report written."


async def review_report(ctx: Context, review: str) -> str:
    """Useful for reviewing a report and providing feedback."""
    current_state = await ctx.get("state")
    current_state["review"] = review
    await ctx.set("state", current_state)
    return "Report reviewed."
```

Now we're going to bring in a new class to create a stand-alone function-calling agent, the `FunctionAgent` (we also support a `ReactAgent`):

```python
from llama_index.core.agent.workflow import FunctionAgent
```

Using it, we'll create the first of our agents, the `ResearchAgent` which will search the web for information using the `search_web` tool and use the `record_notes` tool to save those notes to the state for other agents to use. The key syntactical elements to note here are:
* The `name`, which is used to identify the agent to other agents, as we'll see shortly
* The `description`, which is used by other agents to decide who to hand off control to next
* The `system_prompt`, which defines the behavior of the agent
* `can_handoff_to` is an optional list of agent names that the agent can hand control to. By default, it will be able to hand control to any other agent.

```python
research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Useful for searching the web for information on a given topic and recording notes on the topic.",
    system_prompt=(
        "You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
        "Once notes are recorded and you are satisfied, you should hand off control to the WriteAgent to write a report on the topic."
    ),
    llm=llm,
    tools=[search_web, record_notes],
    can_handoff_to=["WriteAgent"],
)
```

Our other two agents are defined similarly, with different tools and system prompts:

```python
write_agent = FunctionAgent(
    name="WriteAgent",
    description="Useful for writing a report on a given topic.",
    system_prompt=(
        "You are the WriteAgent that can write a report on a given topic. "
        "Your report should be in a markdown format. The content should be grounded in the research notes. "
        "Once the report is written, you should get feedback at least once from the ReviewAgent."
    ),
    llm=llm,
    tools=[write_report],
    can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Useful for reviewing a report and providing feedback.",
    system_prompt=(
        "You are the ReviewAgent that can review a report and provide feedback. "
        "Your feedback should either approve the current report or request changes for the WriteAgent to implement."
    ),
    llm=llm,
    tools=[review_report],
    can_handoff_to=["WriteAgent"],
)
```

With our agents defined, we can now instantiate our `AgentWorkflow` directly to create a multi-agent system. We give it an array of our agents, and define which one should initially have control using `root_agent`. We can also define the initial value of the `state` variable, which as we've [seen previously](./state.md), is a dictionary that can be accessed by all agents.

```python
agent_workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent.name,
    initial_state={
        "research_notes": {},
        "report_content": "Not written yet.",
        "review": "Review required.",
    },
)
```

Now we're ready to run our multi-agent system. We've added some event-handling [using streaming events](./streaming.md) to make it clearer what's happening under the hood:

```python
handler = agent_workflow.run(
    user_msg="""
    Write me a report on the history of the web. Briefly describe the history
    of the world wide web, including the development of the internet and the
    development of the web, including 21st century developments.
"""
)

current_agent = None
current_tool_calls = ""
async for event in handler.stream_events():
    if (
        hasattr(event, "current_agent_name")
        and event.current_agent_name != current_agent
    ):
        current_agent = event.current_agent_name
        print(f"\n{'='*50}")
        print(f"🤖 Agent: {current_agent}")
        print(f"{'='*50}\n")
    elif isinstance(event, AgentOutput):
        if event.response.content:
            print("📤 Output:", event.response.content)
        if event.tool_calls:
            print(
                "🛠️  Planning to use tools:",
                [call.tool_name for call in event.tool_calls],
            )
    elif isinstance(event, ToolCallResult):
        print(f"🔧 Tool Result ({event.tool_name}):")
        print(f"  Arguments: {event.tool_kwargs}")
        print(f"  Output: {event.tool_output}")
    elif isinstance(event, ToolCall):
        print(f"🔨 Calling Tool: {event.tool_name}")
        print(f"  With arguments: {event.tool_kwargs}")
```

This gives us some very verbose output, which we've truncated here for brevity:

```
==================================================
🤖 Agent: ResearchAgent
==================================================

🛠️  Planning to use tools: ['search']
🔨 Calling Tool: search
  With arguments: {'query': 'history of the world wide web and internet development', 'max_results': 6}
🔧 Tool Result (search):
  Arguments: {'query': 'history of the world wide web and internet development', 'max_results': 6}
  Output: [Document(id_='2e977310-2994-4ea9-ade2-8da4533983e8', embedding=None, metadata={'url': 'https://www.scienceandmediamuseum.org.uk/objects-and-stories/short-history-internet'}, excluded_embed_metadata_keys=[], ...
🛠️  Planning to use tools: ['record_notes', 'record_notes']
🔨 Calling Tool: record_notes
  With arguments: {'notes': 'The World Wide Web (WWW) was created by Tim Berners-Lee...','notes_title': 'History of the World Wide Web and Internet Development'}
🔧 Tool Result (record_notes):
  Arguments: {'notes': 'The World Wide Web (WWW) was created by Tim Berners-Lee...', 'notes_title': 'History of the World Wide Web and Internet Development'}
  Output: Notes recorded.
🔨 Calling Tool: record_notes
  With arguments: {'notes': "The internet's origins trace back to the 1950s....", 'notes_title': '21st Century Developments in Web Technology'}
🔧 Tool Result (record_notes):
  Arguments: {'notes': "The internet's origins trace back to the 1950s... .", 'notes_title': '21st Century Developments in Web Technology'}
  Output: Notes recorded.
🛠️  Planning to use tools: ['handoff']
🔨 Calling Tool: handoff
  With arguments: {'to_agent': 'WriteAgent', 'reason': 'I have recorded the necessary notes on the history of the web and its developments.'}
🔧 Tool Result (handoff):
  Arguments: {'to_agent': 'WriteAgent', 'reason': 'I have recorded the necessary notes on the history of the web and its developments.'}
  Output: Agent WriteAgent is now handling the request due to the following reason: I have recorded the necessary notes on the history of the web and its developments..
Please continue with the current request.
```

You can see that `ResearchAgent` has found some notes and handed control to `WriteAgent`, which generates `report_content`:

```
==================================================
🤖 Agent: WriteAgent
==================================================

🛠️  Planning to use tools: ['write_report']
🔨 Calling Tool: write_report
  With arguments: {'report_content': '# History of the World Wide Web...'}
🔧 Tool Result (write_report):
  Arguments: {'report_content': '# History of the World Wide Web...'}
  Output: Report written.
🛠️  Planning to use tools: ['handoff']
🔨 Calling Tool: handoff
  With arguments: {'to_agent': 'ReviewAgent', 'reason': 'The report on the history of the web has been completed and requires review.'}
🔧 Tool Result (handoff):
  Arguments: {'to_agent': 'ReviewAgent', 'reason': 'The report on the history of the web has been completed and requires review.'}
  Output: Agent ReviewAgent is now handling the request due to the following reason: The report on the history of the web has been completed and requires review..
Please continue with the current request.
```

And finally control is passed to the `ReviewAgent` to review the report:

```
==================================================
🤖 Agent: ReviewAgent
==================================================

🛠️  Planning to use tools: ['review_report']
🔨 Calling Tool: review_report
  With arguments: {'review': 'The report on the history of the web is well-structured ... Approval is granted.'}
🔧 Tool Result (review_report):
  Arguments: {'review': 'The report on the history of the web is well-structured ... Approval is granted.'}
  Output: Report reviewed.
📤 Output: The report on the history of the web has been reviewed and approved. It effectively covers the key developments from the inception of the internet to the 21st century, including significant contributions and advancements. If you need any further assistance or additional reports, feel free to ask!
```

You can see the [full code of this example](https://github.com/run-llama/python-agents-tutorial/blob/main/6_multi_agent.py).

As an extension of this example, you could create a system that takes the feedback from the `ReviewAgent` and passes it back to the `WriteAgent` to update the report.

## Congratulations!

You've covered all there is to know about building agents with `AgentWorkflow`. In the [Workflows tutorial](../workflows/index.md), you'll take many of the concepts you've learned here and apply them to building more precise, lower-level agentic systems.
