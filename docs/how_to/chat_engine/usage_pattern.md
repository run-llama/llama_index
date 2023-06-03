# Usage Pattern

We provide a few simple implementations to start, with more sophisticated modes coming soon!
More specifically, the `SimpleChatEngine` does not make use of a knowledge base, 
whereas `CondenseQuestionChatEngine` and `ReActChatEngine` make use of a query engine over knowledge base.

Configuring a chat engine is very similar to configuring a query engine.

## High-Level API
You can directly build a chat engine from an index in 1 line of code:
```python
chat_engine = index.as_chat_engine(
    chat_mode='condense_question', 
    verbose=True
)
```
> Note: you can access different chat engines by specifying the `chat_mode` as a kwarg. `condense_question` corresponds to `CondenseQuestionChatEngine`, `react` corresponds to `ReActChatEngine`.

Call `chat` to have a conversation with your data:
```python
response = chat_engine.chat("Tell me a joke.")
```

Call `reset` to start a new conversation:
```python
chat_engine.reset()
```


## Low-Level Composition API
While the high-level API optimizes for ease-of-use, it does not expose full range of configurability.  

If you need more granular control, you can use the low-level composition API.
Concretely speaking, you would explicitly construct `ChatEngine` object (you may need to look at API references or example notebooks) instead of calling `index.as_chat_engine(...)`.


For example, you might want to:
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