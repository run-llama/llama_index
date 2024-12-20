# LlamaIndex Llms Integration: Llama Cpp

## Installation

1. Install the required Python packages:

   ```bash
   %pip install llama-index-embeddings-huggingface
   %pip install llama-index-llms-llama-cpp
   !pip install llama-index
   ```

## Basic Usage

### Import Required Libraries

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
```

### Initialize LlamaCPP

Set up the model URL and initialize the LlamaCPP LLM:

```python
model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
llm = LlamaCPP(
    model_url=model_url,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 1},
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
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
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

https://docs.llamaindex.ai/en/stable/examples/llm/llama_2_llama_cpp/
