# LlamaIndex Llms Integration: Gemini

**NOTE:** Gemini has largely been replaced by Google GenAI. Visit the [Google GenAI page](https://docs.llamaindex.ai/en/stable/examples/llm/google_genai/) for the latest examples and documentation.

## Installation

1. Install the required Python packages:

   ```bash
   %pip install llama-index-llms-gemini
   !pip install -q llama-index google-generativeai
   ```

2. Set the Google API key as an environment variable:

   ```bash
   %env GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

### Basic Content Generation

To generate a poem using the Gemini model, use the following code:

```python
from llama_index.llms.gemini import Gemini

resp = Gemini().complete("Write a poem about a magic backpack")
print(resp)
```

### Chat with Messages

To simulate a conversation, send a list of messages:

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.gemini import Gemini

messages = [
    ChatMessage(role="user", content="Hello friend!"),
    ChatMessage(role="assistant", content="Yarr what is shakin' matey?"),
    ChatMessage(
        role="user", content="Help me decide what to have for dinner."
    ),
]
resp = Gemini().chat(messages)
print(resp)
```

### Streaming Responses

To stream content responses in real-time:

```python
from llama_index.llms.gemini import Gemini

llm = Gemini()
resp = llm.stream_complete(
    "The story of Sourcrust, the bread creature, is really interesting. It all started when..."
)
for r in resp:
    print(r.text, end="")
```

To stream chat responses:

```python
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage

llm = Gemini()
messages = [
    ChatMessage(role="user", content="Hello friend!"),
    ChatMessage(role="assistant", content="Yarr what is shakin' matey?"),
    ChatMessage(
        role="user", content="Help me decide what to have for dinner."
    ),
]
resp = llm.stream_chat(messages)
```

### Using Other Models

To find suitable models available in the Gemini model site:

```python
import google.generativeai as genai

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
```

### Specific Model Usage

To use a specific model, you can configure it like this:

```python
from llama_index.llms.gemini import Gemini

llm = Gemini(model="models/gemini-pro")
resp = llm.complete("Write a short, but joyous, ode to LlamaIndex")
print(resp)
```

### Asynchronous API

To use the asynchronous completion API:

```python
from llama_index.llms.gemini import Gemini

llm = Gemini()
resp = await llm.acomplete("Llamas are famous for ")
print(resp)
```

For asynchronous streaming of responses:

```python
resp = await llm.astream_complete("Llamas are famous for ")
async for chunk in resp:
    print(chunk.text, end="")
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/gemini/
