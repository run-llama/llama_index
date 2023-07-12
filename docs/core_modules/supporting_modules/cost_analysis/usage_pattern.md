# Usage Pattern

## Estimating LLM and Embedding Token Counts

In order to measure LLM and Embedding token counts, you'll need to

1. Setup `MockLLM` and `MockEmbedding` objects

```python
from llama_index.llms import MockLLM
from llama_index import MockEmbedding

llm = MockLLM(max_tokens=256)
embed_model = MockEmbedding(embed_dim=1536)
```

2. Setup the `TokenCountingCallback` handler

```python
import tiktoken
from llama_index.callbacks import CallbackManager, TokenCountingHandler

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

callback_manager = CallbackManager([token_counter])
```

3. Add them to the global `ServiceContext`

```python
from llama_index import ServiceContext, set_global_service_context

set_global_service_context(
    ServiceContext.from_defaults(
        llm=llm, 
        embed_model=embed_model, 
        callback_manager=callback_manager
    )
)
```

4. Construct an Index 

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./docs/examples/data/paul_graham").load_data()

index = VectorStoreIndex.from_documents(documents)
```

5. Measure the counts!

```python
print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)

# reset counts
token_counter.reset_counts()
```

6. Run a query, mesaure again

```python
query_engine = index.as_query_engine()

response = query_engine.query("query")

print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)
```
