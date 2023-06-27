# Token Counting - Migration Guide

The existing token counting implementation has been __deprecated__. 

We know token counting is important to many users, so this guide was created to walkthrough a (hopefully painless) transition. 

Previously, token counting was kept track of on the `llm_predictor` and `embed_model` objects directly, and optionally printed to the console. This implementation used a static tokenizer for token counting (gpt-2), and the `last_token_usage` and `total_token_usage` attributes were not always kept track of properly.

Going forward, token counting as moved into a callback. Using the `TokenCountingHandler` callback, you now have more options for how tokens are counted, the lifetime of the token counts, and even creating separete token counters for different indexes.

Here is a minimum example of using the new `TokenCountingHandler` with an OpenAI model:

```python
import tiktoken
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

# you can set a tokenizer directly, or optionally let it default 
# to the same tokenizer that was used previously for token counting
# NOTE: The tokenizer should be a function that takes in text and returns a list of tokens
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("text-davinci-003").encode
    verbose=False  # set to true to see usage printed to the console
)

callback_manager = CallbackManager([token_counter])

service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

document = SimpleDirectoryReader("./data").load_data()

# if verbose is turned on, you will see embedding token usage printed
index = VectorStoreIndex.from_documents(documents)

# otherwise, you can access the count directly
print(token_counter.total_embedding_token_count)

# reset the counts at your discretion!
token_counter.reset_counts()

# also track prompt, completion, and total LLM tokens, in addition to embeddings
response = index.as_query_engine().query("What did the author do growing up?")
print('Embedding Tokens: ', token_counter.total_embedding_token_count, '\n',
      'LLM Prompt Tokens: ', token_counter.prompt_llm_token_count, '\n',
      'LLM Completion Tokens: ', token_counter.completion_llm_token_count, '\n',
      'Total LLM Token Count: ', token_counter.total_llm_token_count)
```
