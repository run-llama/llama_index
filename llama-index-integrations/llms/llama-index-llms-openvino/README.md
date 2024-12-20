# LlamaIndex Llms Integration: Openvino

## Installation

To install the required packages, run:

```bash
%pip install llama-index-llms-openvino transformers huggingface_hub
!pip install llama-index
```

## Setup

### Define Functions for Prompt Handling

You will need functions to convert messages and completions into prompts:

```python
from llama_index.llms.openvino import OpenVINOLLM


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

Models can be loaded by specifying parameters using the `OpenVINOLLM` method. If you have an Intel GPU, specify `device_map="gpu"` to run inference on it:

```python
ov_config = {
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
    "CACHE_DIR": "",
}

ov_llm = OpenVINOLLM(
    model_id_or_path="HuggingFaceH4/zephyr-7b-beta",
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"ov_config": ov_config},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="cpu",
)

response = ov_llm.complete("What is the meaning of life?")
print(str(response))
```

### Inference with Local OpenVINO Model

Export your model to the OpenVINO IR format using the CLI and load it from a local folder. It’s recommended to apply 8 or 4-bit weight quantization to reduce inference latency and model footprint:

```bash
!optimum-cli export openvino --model HuggingFaceH4/zephyr-7b-beta ov_model_dir
!optimum-cli export openvino --model HuggingFaceH4/zephyr-7b-beta --weight-format int8 ov_model_dir
!optimum-cli export openvino --model HuggingFaceH4/zephyr-7b-beta --weight-format int4 ov_model_dir
```

You can then load the model from the specified directory:

```python
ov_llm = OpenVINOLLM(
    model_id_or_path="ov_model_dir",
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"ov_config": ov_config},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="gpu",
)
```

### Additional Optimization

You can get additional inference speed improvements with dynamic quantization of activations and KV-cache quantization. Enable these options with `ov_config` as follows:

```python
ov_config = {
    "KV_CACHE_PRECISION": "u8",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
    "CACHE_DIR": "",
}
```

## Streaming Responses

To use the streaming capabilities, you can use the `stream_complete` and `stream_chat` methods:

### Using `stream_complete`

```python
response = ov_llm.stream_complete("Who is Paul Graham?")
for r in response:
    print(r.delta, end="")
```

### Using `stream_chat`

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]

resp = ov_llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/openvino/
