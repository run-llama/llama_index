# LlamaIndex Llms Integration: Google GenAI

## Installation

1. Install the required Python packages:

   ```bash
   %pip install llama-index-llms-google-genai
   ```

2. Set the Google API key as an environment variable:

   ```bash
   %env GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

### Basic Content Generation

To generate a poem using the Gemini model, use the following code:

```python
from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(model="gemini-2.0-flash")
resp = llm.complete("Write a poem about a magic backpack")
print(resp)
```

### Chat with Messages

To simulate a conversation, send a list of messages:

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI

messages = [
    ChatMessage(role="user", content="Hello friend!"),
    ChatMessage(role="assistant", content="Yarr what is shakin' matey?"),
    ChatMessage(
        role="user", content="Help me decide what to have for dinner."
    ),
]

llm = GoogleGenAI(model="gemini-2.0-flash")
resp = llm.chat(messages)
print(resp)
```

### Streaming Responses

To stream content responses in real-time:

```python
from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(model="gemini-2.0-flash")
resp = llm.stream_complete(
    "The story of Sourcrust, the bread creature, is really interesting. It all started when..."
)
for r in resp:
    print(r.text, end="")
```

To stream chat responses:

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(model="gemini-2.0-flash")
messages = [
    ChatMessage(role="user", content="Hello friend!"),
    ChatMessage(role="assistant", content="Yarr what is shakin' matey?"),
    ChatMessage(
        role="user", content="Help me decide what to have for dinner."
    ),
]
resp = llm.stream_chat(messages)
```

### Specific Model Usage

To use a specific model, you can configure it like this:

```python
from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(model="models/gemini-pro")
resp = llm.complete("Write a short, but joyous, ode to LlamaIndex")
print(resp)
```

### Asynchronous API

To use the asynchronous completion API:

```python
from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(model="models/gemini-pro")
resp = await llm.acomplete("Llamas are famous for ")
print(resp)
```

For asynchronous streaming of responses:

```python
resp = await llm.astream_complete("Llamas are famous for ")
async for chunk in resp:
    print(chunk.text, end="")
```
