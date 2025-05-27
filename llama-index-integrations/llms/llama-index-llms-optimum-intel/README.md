# LlamaIndex Llms Integration: Optimum Intel IPEX backend

## Installation

To install the required packages, run:

```bash
%pip install llama-index-llms-optimum-intel
!pip install llama-index
```

## Setup

### Define Functions for Prompt Handling

You will need functions to convert messages and completions into prompts:

```python
from llama_index.llms.optimum_intel import OptimumIntelLLM


def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # Ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # Add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt


def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"
```

### Model Loading

Models can be loaded by specifying parameters using the `OptimumIntelLLM` method:

```python
oi_llm = OptimumIntelLLM(
    model_name="Intel/neural-chat-7b-v3-3",
    tokenizer_name="Intel/neural-chat-7b-v3-3",
    context_window=3900,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="cpu",
)

response = oi_llm.complete("What is the meaning of life?")
print(str(response))
```

### Streaming Responses

To use the streaming capabilities, you can use the `stream_complete` and `stream_chat` methods:

#### Using `stream_complete`

```python
response = oi_llm.stream_complete("Who is Mother Teresa?")
for r in response:
    print(r.delta, end="")
```

#### Using `stream_chat`

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system",
        content="You are an American chef in a small restaurant in New Orleans",
    ),
    ChatMessage(role="user", content="What is your dish of the day?"),
]

resp = oi_llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/optimum_intel/
