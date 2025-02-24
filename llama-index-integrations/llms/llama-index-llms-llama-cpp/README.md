# LlamaIndex Llms Integration: Llama Cpp

## Installation

To get the best performance out of `LlamaCPP`, it is recommended to install the package so that it is compiled with GPU support. A full guide for installing this way is [here](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal).

Full MACOS instructions are also [here](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/).

In general:

- Use `CuBLAS` if you have CUDA and an NVidia GPU
- Use `METAL` if you are running on an M1/M2 MacBook
- Use `CLBLAST` if you are running on an AMD/Intel GPU

Them, install the required llama-index packages:

```bash
pip install llama-index-embeddings-huggingface
pip install llama-index-llms-llama-cpp
```

## Basic Usage

### Initialize LlamaCPP

Set up the model URL and initialize the LlamaCPP LLM:

```python
from llama_index.llms.llama_cpp import LlamaCPP
from transformers import AutoTokenizer

model_url = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q3_k_m.gguf"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def messages_to_prompt(messages):
    messages = [{"role": m.role.value, "content": m.content} for m in messages]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def completion_to_prompt(completion):
    messages = [{"role": "user", "content": completion}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=16384,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)
```

### Generate Completions

Use the `complete` method to generate a response:

```python
response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
print(response.text)
```

### Stream Completions

You can also stream completions for a prompt:

```python
response_iter = llm.stream_complete("Can you write me a poem about fast cars?")
for response in response_iter:
    print(response.delta, end="", flush=True)
```

### Set Up Query Engine with LlamaCPP

Change the global tokenizer to match the LLM:

```python
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer

set_global_tokenizer(
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct").encode
)
```

### Use Hugging Face Embeddings

Set up the embedding model and load documents:

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
documents = SimpleDirectoryReader(
    "../../../examples/paul_graham_essay/data"
).load_data()
```

### Create Vector Store Index

Create a vector store index from the loaded documents:

```python
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
```

### Set Up Query Engine

Set up the query engine with the LlamaCPP LLM:

```python
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What did the author do growing up?")
print(response)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/llama_cpp/
