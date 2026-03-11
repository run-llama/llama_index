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
from llama_index.core.base.llms.types import ChatMessage
# Set your API key
api_key = "Your api key"
SAMPLE_MODEL = 'command-r-08-2024'

llm = Cohere(model=SAMPLE_MODEL,api_key=api_key)

#Call chat function
resp = llm.chat(messages= [ChatMessage(content="Who is Paul Graham?")])
print(resp)

# Output
# an English computer scientist, entrepreneur and investor.
# He is best known for his work as a co-founder of the seed accelerator Y Combinator.
# He is also the author of the free startup advice blog "Startups.com".
# Paul Graham is known for his philanthropic efforts.
# Has given away hundreds of millions of dollars to good causes.

# Call chat with a list of messages
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
# Import Cohere
from llama_index.llms.cohere import Cohere
from llama_index.core.base.llms.types import ChatMessage

llm = Cohere(model=SAMPLE_MODEL,api_key=api_key)

resp = llm.stream_complete("Paul Graham is")
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
SAMPLE_MODEL = 'command-r-08-2024'

# Import Cohere
from llama_index.llms.cohere import Cohere
from llama_index.core.base.llms.types import ChatMessage

llm = Cohere(model=SAMPLE_MODEL,api_key=api_key)

resp = llm.chat(messages= [ChatMessage(content="Who is Paul Graham?")])
# Note: Your text contains a trailing whitespace, which has been trimmed to ensure high quality generations.
print(resp)

# Output
# an English computer scientist, entrepreneur and investor.
# He is best known for his work as a co-founder of the seed accelerator Y Combinator.
# He is also the co-founder of the online dating platform Match.com.

# Async calls
resp = await llm.acomplete("Paul Graham is")
# Note: Your text contains a trailing whitespace, which has been trimmed to ensure high quality generations.
print(resp)

# Output
# an English computer scientist, entrepreneur and investor.
# He is best known for his work as a co-founder of the startup incubator and seed fund
# Y Combinator, and the programming language Lisp. He has also written numerous essays,
# many of which have become highly influential in the software engineering field.

# Streaming async
resp = await llm.astream_complete("Paul Graham is")
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
from llama_index.core.base.llms.types import ChatMessage

SAMPLE_MODEL = 'command-r-08-2024'

llm_good = Cohere(api_key=api_key)
llm_bad = Cohere(api_key="BAD_KEY")

resp = llm_good.chat(messages= [ChatMessage(content="Who is Paul Graham?")])
print(resp)

resp = llm_bad.chat(messages= [ChatMessage(content="Who is Paul Graham?")])
print(resp)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/cohere/

### Using a Custom Base URL

You can now specify a custom base URL when initializing the Cohere LLM. This is useful for enterprise scenarios or when using a proxy.

```python
from llama_index.llms.cohere import Cohere
from llama_index.core.base.llms.types import ChatMessage

# Initialize with a custom base URL
llm = Cohere(
    api_key="your-api-key", base_url="https://your-custom-endpoint.com/v1"
)

resp = llm.complete("What is LlamaIndex?")
print(resp)
```
