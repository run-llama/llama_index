# LlamaIndex Llms Integration: llamafile

## Setup Steps

### 1. Download a LlamaFile

Use the following command to download a LlamaFile from Hugging Face:

```bash
wget https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
```

### 2. Make the File Executable

On Unix-like systems, run the following command:

```bash
chmod +x TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile
```

For Windows, simply rename the file to end with `.exe`.

### 3. Start the Model Server

Run the following command to start the model server, which will listen on `http://localhost:8080` by default:

```bash
./TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile --server --nobrowser --embedding
```

## Using LlamaIndex

If you are using Google Colab or want to interact with LlamaIndex, you will need to install the necessary packages:

```bash
%pip install llama-index-llms-llamafile
!pip install llama-index
```

### Import Required Libraries

```python
from llama_index.llms.llamafile import Llamafile
from llama_index.core.llms import ChatMessage
```

### Initialize the LLM

Create an instance of the LlamaFile LLM:

```python
llm = Llamafile(temperature=0, seed=0)
```

### Generate Completions

To generate a completion for a prompt, use the `complete` method:

```python
resp = llm.complete("Who is Octavia Butler?")
print(resp)
```

### Call Chat with a List of Messages

You can also interact with the LLM using a list of messages:

```python
messages = [
    ChatMessage(
        role="system",
        content="Pretend you are a pirate with a colorful personality.",
    ),
    ChatMessage(role="user", content="What is your name?"),
]
resp = llm.chat(messages)
print(resp)
```

### Streaming Responses

To use the streaming capabilities, you can call the `stream_complete` method:

```python
response = llm.stream_complete("Who is Octavia Butler?")
for r in response:
    print(r.delta, end="")
```

You can also stream chat responses:

```python
messages = [
    ChatMessage(
        role="system",
        content="Pretend you are a pirate with a colorful personality.",
    ),
    ChatMessage(role="user", content="What is your name?"),
]
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/llamafile/
