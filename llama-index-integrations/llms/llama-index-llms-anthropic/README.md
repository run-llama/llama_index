# LlamaIndex LLM Integration: Anthropic

Anthropic is an AI research company focused on developing advanced language models, notably the Claude series. Their flagship model, Claude, is designed to generate human-like text while prioritizing safety and alignment with human intentions. Anthropic aims to create AI systems that are not only powerful but also responsible, addressing potential risks associated with artificial intelligence.

### Installation

```sh
%pip install llama-index-llms-anthropic
!pip install llama-index
```

```
# Set Tokenizer
# First we want to set the tokenizer, which is slightly different than TikToken.
# NOTE: The Claude 3 tokenizer has not been updated yet; using the existing Anthropic tokenizer leads
# to context overflow errors for 200k tokens. We've temporarily set the max tokens for Claude 3 to 180k.
```

### Basic Usage

```py
import os

from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings


os.environ["ANTHROPIC_API_KEY"] = "YOUR ANTHROPIC API KEY"
from llama_index.llms.anthropic import Anthropic

# To customize your API key, do this
# otherwise it will lookup ANTHROPIC_API_KEY from your env variable
# llm = Anthropic(api_key="<api_key>")
llm = Anthropic(model="claude-3-opus-20240229")

Settings.tokenizer = llm.tokenizer

resp = llm.complete("Paul Graham is ")
print(resp)

# Sample response
# Paul Graham is a well-known entrepreneur, programmer, venture capitalist, and essayist.
# He is best known for co-founding Viaweb, one of the first web application companies, which was later
# sold to Yahoo! in 1998 and became Yahoo! Store. Graham is also the co-founder of Y Combinator, a highly
# successful startup accelerator that has helped launch numerous successful companies, such as Dropbox,
# Airbnb, and Reddit.
```

### Using Anthropic model through Vertex AI

```py
import os

os.environ["ANTHROPIC_PROJECT_ID"] = "YOUR PROJECT ID HERE"
os.environ["ANTHROPIC_REGION"] = "YOUR PROJECT REGION HERE"
# Set region and project_id to make Anthropic use the Vertex AI client

llm = Anthropic(
    model="claude-3-5-sonnet@20240620",
    region=os.getenv("ANTHROPIC_REGION"),
    project_id=os.getenv("ANTHROPIC_PROJECT_ID"),
)

resp = llm.complete("Paul Graham is ")
print(resp)
```

### Chat example with a list of messages

```py
from llama_index.core.llms import ChatMessage
from llama_index.llms.anthropic import Anthropic

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = Anthropic(model="claude-3-opus-20240229").chat(messages)
print(resp)
```

### Streaming example

```py
from llama_index.llms.anthropic import Anthropic

llm = Anthropic(model="claude-3-opus-20240229", max_tokens=100)
resp = llm.stream_complete("Paul Graham is ")
for r in resp:
    print(r.delta, end="")
```

### Chat streaming with pirate story

```py
llm = Anthropic(model="claude-3-opus-20240229")
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
```

### Configure Model

```py
from llama_index.llms.anthropic import Anthropic

llm = Anthropic(model="claude-3-sonnet-20240229")
resp = llm.stream_complete("Paul Graham is ")
for r in resp:
    print(r.delta, end="")
```

### Async completion

```py
from llama_index.llms.anthropic import Anthropic

llm = Anthropic("claude-3-sonnet-20240229")
resp = await llm.acomplete("Paul Graham is ")
print(resp)
```

### Using Anthropic Tools (Web Search)

```py
from llama_index.llms.anthropic import Anthropic

# Initialize with web search tool
llm = Anthropic(
    model="claude-3-7-sonnet-latest",  # Must be a tool-supported model
    max_tokens=1024,
    tools=[
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 3,  # Limit to 3 searches
        }
    ],
)

# Get response with citations
response = llm.complete("What are the latest AI research trends?")

# Access the main response content
print(response.text)

# Access citations if available
for citation in response.citations:
    print(f"Source: {citation.get('url')} - {citation.get('cited_text')}")
```

### Structured Prediction Example

```py
from llama_index.llms.anthropic import Anthropic
from llama_index.core.prompts import PromptTemplate
from llama_index.core.bridge.pydantic import BaseModel
from typing import List


class MenuItem(BaseModel):
    """A menu item in a restaurant."""

    course_name: str
    is_vegetarian: bool


class Restaurant(BaseModel):
    """A restaurant with name, city, and cuisine."""

    name: str
    city: str
    cuisine: str
    menu_items: List[MenuItem]


llm = Anthropic("claude-3-5-sonnet-20240620")
prompt_tmpl = PromptTemplate(
    "Generate a restaurant in a given city {city_name}"
)

# Option 1: Use `as_structured_llm`
restaurant_obj = (
    llm.as_structured_llm(Restaurant)
    .complete(prompt_tmpl.format(city_name="Miami"))
    .raw
)
print(restaurant_obj)

# Option 2: Use `structured_predict`
# restaurant_obj = llm.structured_predict(Restaurant, prompt_tmpl, city_name="Miami")

# Streaming Structured Prediction
from llama_index.core.llms import ChatMessage
from IPython.display import clear_output
from pprint import pprint

input_msg = ChatMessage.from_str("Generate a restaurant in San Francisco")

sllm = llm.as_structured_llm(Restaurant)
stream_output = sllm.stream_chat([input_msg])
for partial_output in stream_output:
    clear_output(wait=True)
    pprint(partial_output.raw.dict())
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/anthropic/
