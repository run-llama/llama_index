# Maintaining state

By default, the `AgentWorkflow` is stateless between runs. This means that the agent will not have any memory of previous runs.

To maintain state, we need to keep track of the previous state. In LlamaIndex, Workflows have a `Context` class that can be used to maintain state within and between runs. Since the AgentWorkflow is just a pre-built Workflow, we can also use it now.

```python
from llama_index.core.workflow import Context
```

To maintain state between runs, we'll create a new Context called ctx. We pass in our workflow to properly configure this Context object for the workflow that will use it.

```python
ctx = Context(workflow)
```

With our configured Context, we can pass it to our first run.

```python
response = await workflow.run(user_msg="Hi, my name is Laurie!", ctx=ctx)
print(response)
```

Which gives us:

```
Hello Laurie! How can I assist you today?
```

And now if we run the workflow again to ask a follow-up question, it will remember that information:

```python
response2 = await workflow.run(user_msg="What's my name?", ctx=ctx)
print(response2)
```

Which gives us:

```
Your name is Laurie!
```

## Maintaining state over longer periods

The Context is serializable, so it can be saved to a database, file, etc. and loaded back in later.

The JsonSerializer is a simple serializer that uses `json.dumps` and `json.loads` to serialize and deserialize the context.

The JsonPickleSerializer is a serializer that uses pickle to serialize and deserialize the context. If you have objects in your context that are not serializable, you can use this serializer.

We bring in our serializers as any other import:

```python
from llama_index.core.workflow import JsonPickleSerializer, JsonSerializer
```

We can then serialize our context to a dictionary and save it to a file:

```python
ctx_dict = ctx.to_dict(serializer=JsonSerializer())
```

We can deserialize it back into a Context object and ask questions just as before:

```python
restored_ctx = Context.from_dict(
    workflow, ctx_dict, serializer=JsonSerializer()
)

response3 = await workflow.run(user_msg="What's my name?", ctx=restored_ctx)
```

You can see the [full code of this example](https://github.com/run-llama/python-agents-tutorial/blob/main/3_state.py).

## Tools and state

Tools can also be defined that have access to the workflow context. This means you can set and retrieve variables from the context and use them in the tool, or to pass information between tools.

`AgentWorkflow` uses a context variable called `state` that is available to every agent. You can rely on information in `state` being available without explicitly having to pass it in.

To access the Context, the Context parameter should be the first parameter of the tool, as we're doing here, in a tool that simply adds a name to the state:

```python
async def set_name(ctx: Context, name: str) -> str:
    state = await ctx.get("state")
    state["name"] = name
    await ctx.set("state", state)
    return f"Name set to {name}"
```

We can now create an agent that uses this tool. You can optionally provide the initial state of the agent, which we'll do here:

```python
workflow = AgentWorkflow.from_tools_or_functions(
    [set_name],
    llm=llm,
    system_prompt="You are a helpful assistant that can set a name.",
    initial_state={"name": "unset"},
)
```

Now we can create a Context and ask the agent about the state:

```python
ctx = Context(workflow)

# check if it knows a name before setting it
response = await workflow.run(user_msg="What's my name?", ctx=ctx)
print(str(response))
```

Which gives us:

```
Your name has been set to "unset."
```

Then we can explicitly set the name in a new run of the agent:

```python
response2 = await workflow.run(user_msg="My name is Laurie", ctx=ctx)
print(str(response2))
```

```
Your name has been updated to "Laurie."
```

We could now ask the agent the name again, or we can access the value of the state directly:

```python
state = await ctx.get("state")
print("Name as stored in state: ", state["name"])
```

Which gives us:

```
Name as stored in state: Laurie
```

You can see the [full code of this example](https://github.com/run-llama/python-agents-tutorial/blob/main/3a_tools_and_state.py).

Next we'll learn about [streaming output and events](./streaming.md).
