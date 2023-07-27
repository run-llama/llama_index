# Usage Pattern

## Get Started

Build a chat engine from index:
```python
chat_engine = index.as_chat_engine()
```

```{tip}
To learn how to build an index, see [Index](/core_modules/data_modules/index/root.md)
```

Have a conversation with your data:
```python
response = chat_engine.chat("Tell me a joke.")
```

Reset chat history to start a new conversation:
```python
chat_engine.reset()
```

Enter an interactive chat REPL:
```python
chat_engine.chat_repl()
```


## Configuring a Chat Engine
Configuring a chat engine is very similar to configuring a query engine.

### High-Level API
You can directly build and configure a chat engine from an index in 1 line of code:
```python
chat_engine = index.as_chat_engine(
    chat_mode='condense_question', 
    verbose=True
)
```
> Note: you can access different chat engines by specifying the `chat_mode` as a kwarg. `condense_question` corresponds to `CondenseQuestionChatEngine`, `react` corresponds to `ReActChatEngine`, `context` corresponds to a `ContextChatEngine`.

> Note: While the high-level API optimizes for ease-of-use, it does *NOT* expose full range of configurability.  

#### Available Chat Modes

- `best` - Turn the query engine into a tool, for use with a `ReAct` data agent or an `OpenAI` data agent, depending on what your LLM supports. `OpenAI` data agents require `gpt-3.5-turbo` or `gpt-4` as they use the function calling API from OpenAI.
- `context` - Retrieve nodes from the index using every user message. The retrieved text is inserted into the system prompt, so that the chat engine can either respond naturally or use the context from the query engine.
- `condense_question` - Look at the chat history and re-write the user message to be a query for the index. Return the response after reading the response from the query engine.
- `simple` - A simple chat with the LLM directly, no query engine involved.
- `react` - Same as `best`, but forces a `ReAct` data agent.
- `openai` - Same as `best`, but forces an `OpenAI` data agent.

### Low-Level Composition API

You can use the low-level composition API if you need more granular control.
Concretely speaking, you would explicitly construct `ChatEngine` object instead of calling `index.as_chat_engine(...)`.
> Note: You may need to look at API references or example notebooks.

Here's an example where we configure the following:
* configure the condense question prompt, 
* initialize the conversation with some existing history,
* print verbose debug message.

```python
from llama_index.prompts  import Prompt

custom_prompt = Prompt("""\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History> 
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
""")

# list of (human_message, ai_message) tuples
custom_chat_history = [
    (
        'Hello assistant, we are having a insightful discussion about Paul Graham today.', 
        'Okay, sounds good.'
    )
]

query_engine = index.as_query_engine()
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine, 
    condense_question_prompt=custom_prompt,
    chat_history=custom_chat_history,
    verbose=True
)
```

### Streaming
To enable streaming, you simply need to call the `stream_chat` endpoint instead of the `chat` endpoint. 

```{warning}
This somewhat inconsistent with query engine (where you pass in a `streaming=True` flag). We are working on making the behavior more consistent! 
```

```python
chat_engine = index.as_chat_engine()
streaming_response = chat_engine.stream_chat("Tell me a joke.")
for token in streaming_response.response_gen:
    print(token, end="")
```

See an [end-to-end tutorial](/examples/customization/streaming/chat_engine_condense_question_stream_response.ipynb)



