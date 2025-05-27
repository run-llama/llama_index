# LlamaIndex Llms Integration: Cohere

### Installation

```bash
%pip install llama-index-llms-openai
%pip install llama-index-llms-cohere
!pip install llama-index
```

### Basic usage

```py
# Import Cohere
from llama_index.llms.cohere import Cohere

# Set your API key
api_key = "Your api key"

# Call complete function
resp = Cohere(api_key=api_key).complete("Paul Graham is ")
# Note: Your text contains a trailing whitespace, which has been trimmed to ensure high quality generations.
print(resp)

# Output
# an English computer scientist, entrepreneur and investor.
# He is best known for his work as a co-founder of the seed accelerator Y Combinator.
# He is also the author of the free startup advice blog "Startups.com".
# Paul Graham is known for his philanthropic efforts.
# Has given away hundreds of millions of dollars to good causes.

# Call chat with a list of messages
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="user", content="hello there"),
    ChatMessage(
        role="assistant", content="Arrrr, matey! How can I help ye today?"
    ),
    ChatMessage(role="user", content="What is your name"),
]

resp = Cohere(api_key=api_key).chat(
    messages, preamble_override="You are a pirate with a colorful personality"
)
print(resp)

# Output
# assistant: Traditionally, ye refers to gender-nonconforming people of any gender,
# and those who are genderless, whereas matey refers to a friend, commonly used to
# address a fellow pirate. According to pop culture in works like "Pirates of the
# Caribbean", the romantic interest of Jack Sparrow refers to themselves using the
# gender-neutral pronoun "ye".

# Are you interested in learning more about the pirate culture?
```

### Streaming: Using stream_complete endpoint

```py
from llama_index.llms.cohere import Cohere

llm = Cohere(api_key=api_key)
resp = llm.stream_complete("Paul Graham is ")
for r in resp:
    print(r.delta, end="")

# Output
# an English computer scientist, essayist, and venture capitalist.
# He is best known for his work as a co-founder of the Y Combinator startup incubator,
# and his essays, which are widely read and influential in the startup community.

# Using stream_chat endpoint
messages = [
    ChatMessage(role="user", content="hello there"),
    ChatMessage(
        role="assistant", content="Arrrr, matey! How can I help ye today?"
    ),
    ChatMessage(role="user", content="What is your name"),
]

resp = llm.stream_chat(
    messages, preamble_override="You are a pirate with a colorful personality"
)
for r in resp:
    print(r.delta, end="")

# Output
# Arrrr, matey! According to etiquette, we are suppose to exchange names first!
# Mine remains a mystery for now.
```

### Configure Model

```py
llm = Cohere(model="command", api_key=api_key)
resp = llm.complete("Paul Graham is ")
# Note: Your text contains a trailing whitespace, which has been trimmed to ensure high quality generations.
print(resp)

# Output
# an English computer scientist, entrepreneur and investor.
# He is best known for his work as a co-founder of the seed accelerator Y Combinator.
# He is also the co-founder of the online dating platform Match.com.

# Async calls
llm = Cohere(model="command", api_key=api_key)
resp = await llm.acomplete("Paul Graham is ")
# Note: Your text contains a trailing whitespace, which has been trimmed to ensure high quality generations.
print(resp)

# Output
# an English computer scientist, entrepreneur and investor.
# He is best known for his work as a co-founder of the startup incubator and seed fund
# Y Combinator, and the programming language Lisp. He has also written numerous essays,
# many of which have become highly influential in the software engineering field.

# Streaming async
resp = await llm.astream_complete("Paul Graham is ")
async for delta in resp:
    print(delta.delta, end="")

# Output
# an English computer scientist, essayist, and businessman.
# He is best known for his work as a co-founder of the startup accelerator Y Combinator,
# and his essay "Beating the Averages."
```

### Set API Key at a per-instance level

```py
# If desired, you can have separate LLM instances use separate API keys.
from llama_index.llms.cohere import Cohere

llm_good = Cohere(api_key=api_key)
llm_bad = Cohere(model="command", api_key="BAD_KEY")

resp = llm_good.complete("Paul Graham is ")
print(resp)

resp = llm_bad.complete("Paul Graham is ")
print(resp)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/cohere/
