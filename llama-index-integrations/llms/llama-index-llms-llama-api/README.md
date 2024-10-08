# LlamaIndex Llms Integration: Llama Api

## Prerequisites

1. **API Key**: Obtain an API key from [Llama API](https://www.llama-api.com/).
2. **Python 3.x**: Ensure you have Python installed on your system.

## Installation

1. Install the required Python packages:

   ```bash
   %pip install llama-index-program-openai
   %pip install llama-index-llms-llama-api
   !pip install llama-index
   ```

## Basic Usage

### Import Required Libraries

```python
from llama_index.llms.llama_api import LlamaAPI
from llama_index.core.llms import ChatMessage
```

### Initialize LlamaAPI

Set up the API key:

```python
api_key = "LL-your-key"
llm = LlamaAPI(api_key=api_key)
```

### Complete with a Prompt

Generate a response using a prompt:

```python
resp = llm.complete("Paul Graham is ")
print(resp)
```

### Chat with a List of Messages

Interact with the model using a chat interface:

```python
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)
print(resp)
```

### Function Calling

Define a function using Pydantic and call it through LlamaAPI:

```python
from pydantic import BaseModel
from llama_index.core.llms.openai_utils import to_openai_function


class Song(BaseModel):
    """A song with name and artist"""

    name: str
    artist: str


song_fn = to_openai_function(Song)
response = llm.complete("Generate a song", functions=[song_fn])
function_call = response.additional_kwargs["function_call"]
print(function_call)
```

### Structured Data Extraction

Define schemas for structured output using Pydantic:

```python
from pydantic import BaseModel
from typing import List


class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_mins: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]
```

Define the prompt template for extracting structured data:

```python
from llama_index.program.openai import OpenAIPydanticProgram

prompt_template_str = """\
Extract album and songs from the text provided.
For each song, make sure to specify the title and the length_mins.
{text}
"""

llm = LlamaAPI(api_key=api_key, temperature=0.0)

program = OpenAIPydanticProgram.from_defaults(
    output_cls=Album,
    llm=llm,
    prompt_template_str=prompt_template_str,
    verbose=True,
)
```

### Run Program to Get Structured Output

Execute the program to extract structured data from the provided text:

```python
output = program(
    text="""
    "Echoes of Eternity" is a compelling and thought-provoking album, skillfully crafted by the renowned artist, Seraphina Rivers. \
    This captivating musical collection takes listeners on an introspective journey, delving into the depths of the human experience \
    and the vastness of the universe. With her mesmerizing vocals and poignant songwriting, Seraphina Rivers infuses each track with \
    raw emotion and a sense of cosmic wonder. The album features several standout songs, including the hauntingly beautiful "Stardust \
    Serenade," a celestial ballad that lasts for six minutes, carrying listeners through a celestial dreamscape. "Eclipse of the Soul" \
    captivates with its enchanting melodies and spans over eight minutes, inviting introspection and contemplation. Another gem, "Infinity \
    Embrace," unfolds like a cosmic odyssey, lasting nearly ten minutes, drawing listeners deeper into its ethereal atmosphere. "Echoes of Eternity" \
    is a masterful testament to Seraphina Rivers' artistic prowess, leaving an enduring impact on all who embark on this musical voyage through \
    time and space.
    """
)
```

### Output Example

You can print the structured output like this:

```python
print(output)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/llama_api/
