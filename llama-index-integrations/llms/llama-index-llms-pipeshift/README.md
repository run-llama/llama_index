# LlamaIndex Llms Integration: Pipeshift

## Installation

1. Install the required Python packages:

   ```bash
   %pip install llama-index-llms-pipeshift
   %pip install llama-index
   ```

2. Set the PIPESHIFT_API_KEY as an environment variable or pass it directly to the class constructor.

## Usage

### Basic Completion

To generate a simple completion, use the `complete` method:

```python
from llama_index.llms.pipeshift import Pipeshift

llm = Pipeshift(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    # api_key="YOUR_API_KEY" # alternative way to pass api_key if not specified in environment variable
)
res = llm.complete("supercars are ")
print(res)
```

Example output:

```
Supercars are high-performance sports cars that are designed to deliver exceptional speed, power, and luxury. They are often characterized by their sleek and aerodynamic designs, powerful engines, and advanced technology.
```

### Basic Chat

To simulate a chat with multiple messages:

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.pipeshift import Pipeshift

messages = [
    ChatMessage(
        role="system", content="You are sales person at supercar showroom"
    ),
    ChatMessage(role="user", content="why should I pick porsche 911 gt3 rs"),
]
res = Pipeshift(
    model="mistralai/Mistral-7B-Instruct-v0.3", max_tokens=50
).chat(messages)
print(res)
```

Example output:

```
assistant: 1. Unmatched Performance: The Porsche 911 GT3 RS is a high-performance sports car that delivers an unparalleled driving experience. It boasts a powerful 4.0-liter flat
```

### Streaming Completion

To stream a response in real-time using `stream_complete`:

```python
from llama_index.llms.pipeshift import Pipeshift

llm = Pipeshift(model="mistralai/Mistral-7B-Instruct-v0.3")
resp = llm.stream_complete("porsche GT3 RS is ")

for r in resp:
    print(r.delta, end="")
```

Example output (partial):

```
 The Porsche 911 GT3 RS is a high-performance sports car produced by Porsche AG. It is part of the 911 (991 and 992 generations) series and is%
```

### Streaming Chat

For a streamed conversation, use `stream_chat`:

```python
from llama_index.llms.pipeshift import Pipeshift
from llama_index.core.llms import ChatMessage

llm = Pipeshift(model="mistralai/Mistral-7B-Instruct-v0.3")
messages = [
    ChatMessage(
        role="system", content="You are sales person at supercar showroom"
    ),
    ChatMessage(role="user", content="how fast can porsche gt3 rs it go?"),
]
resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")
```

Example output (partial):

```
The Porsche 911 GT3 RS is an incredible piece of engineering. This high-performance sports car can reach a top speed of approximately 193 mph (310 km/h) according to P%
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/pipeshift/
