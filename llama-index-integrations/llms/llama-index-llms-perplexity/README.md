# LlamaIndex Llms Integration: Perplexity

## Installation

To install the required packages, run:

```bash
%pip install llama-index-llms-perplexity
!pip install llama-index
```

## Setup

### Import Libraries and Configure API Key

Import the necessary libraries and set your Perplexity API key:

```python
from llama_index.llms.perplexity import Perplexity

pplx_api_key = "your-perplexity-api-key"  # Replace with your actual API key
```

### Initialize the Perplexity LLM

Create an instance of the Perplexity LLM with your API key and desired model settings:

```python
llm = Perplexity(
    api_key=pplx_api_key, model="mistral-7b-instruct", temperature=0.5
)
```

## Chat Example

### Sending a Chat Message

You can send a chat message using the `chat` method. Here’s how to do that:

```python
from llama_index.core.llms import ChatMessage

messages_dict = [
    {"role": "system", "content": "Be precise and concise."},
    {"role": "user", "content": "Tell me 5 sentences about Perplexity."},
]

messages = [ChatMessage(**msg) for msg in messages_dict]

# Get response from the model
response = llm.chat(messages)
print(response)
```

### Async Chat

To send messages asynchronously, you can use the `achat` method:

```python
response = await llm.achat(messages)
print(response)
```

### Stream Chat

For streaming responses, you can use the `stream_chat` method:

```python
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
```

### Async Stream Chat

To stream responses asynchronously, use the `astream_chat` method:

```python
resp = await llm.astream_chat(messages)
async for delta in resp:
    print(delta.delta, end="")
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/perplexity/
