# ServiceContext

## Concept
The `ServiceContext` is a bundle of commonly used resources used during the indexing and querying stage in a LlamaIndex pipeline/application.
You can use it to set the [global configuration](#setting-global-configuration), as well as [local configurations](#setting-local-configuration) at specific parts of the pipeline.

## Usage Pattern

### Configuring the service context
The `ServiceContext` is a simple python dataclass that you can directly construct by passing in the desired components.

```python
@dataclass
class ServiceContext:
    # The LLM used to generate natural language responses to queries.
    # If not provided, defaults to gpt-3.5-turbo from OpenAI
    # If your OpenAI key is not set, defaults to llama2-chat-13B from Llama.cpp
    llm: LLM

    # The PromptHelper object that helps with truncating and repacking text chunks to fit in the LLM's context window.
    prompt_helper: PromptHelper

    # The embedding model used to generate vector representations of text.
    # If not provided, defaults to text-embedding-ada-002
    # If your OpenAI key is not set, defaults to BAAI/bge-small-en
    embed_model: BaseEmbedding

    # The parser that converts documents into nodes.
    node_parser: NodeParser

    # The callback manager object that calls it's handlers on events. Provides basic logging and tracing capabilities.
    callback_manager: CallbackManager

    @classmethod
    def from_defaults(cls, ...) -> "ServiceContext":
      ... 
```

```{tip}
Learn how to configure specific modules:
- [LLM](/core_modules/model_modules/llms/usage_custom.md)
- [Embedding Model](/core_modules/model_modules/embeddings/usage_pattern.md)
- [Node Parser](/core_modules/data_modules/node_parsers/usage_pattern.md)

```

We also expose some common kwargs (of the above components) via the `ServiceContext.from_defaults` method
for convenience (so you don't have to manually construct them).
 
**Kwargs for node parser**:
- `chunk_size`: The size of the text chunk for a node . Is used for the node parser when they aren't provided.
- `chunk_overlap`: The amount of overlap between nodes (i.e. text chunks).

**Kwargs for prompt helper**:
- `context_window`: The size of the context window of the LLM. Typically we set this 
  automatically with the model metadata. But we also allow explicit override via this parameter
  for additional control (or in case the default is not available for certain latest
  models)
- `num_output`: The number of maximum output from the LLM. Typically we set this
  automatically given the model metadata. This parameter does not actually limit the model
  output, it affects the amount of "space" we save for the output, when computing 
  available context window size for packing text from retrieved Nodes.

Here's a complete example that sets up all objects using their default settings:

```python
from llama_index import ServiceContext, LLMPredictor, OpenAIEmbedding, PromptHelper
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser

llm = OpenAI(model='text-davinci-003', temperature=0, max_tokens=256)
embed_model = OpenAIEmbedding()
node_parser = SimpleNodeParser.from_defaults(
  text_splitter=TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
)
prompt_helper = PromptHelper(
  context_window=4096, 
  num_output=256, 
  chunk_overlap_ratio=0.1, 
  chunk_size_limit=None
)

service_context = ServiceContext.from_defaults(
  llm=llm,
  embed_model=embed_model,
  node_parser=node_parser,
  prompt_helper=prompt_helper
)
```

### Setting global configuration
You can set a service context as the global default that applies to the entire LlamaIndex pipeline:

```python
from llama_index import set_global_service_context
set_global_service_context(service_context)
```

### Setting local configuration
You can pass in a service context to specific part of the pipeline to override the default configuration: 

```python
query_engine = index.as_query_engine(service_context=service_context)
response = query_engine.query("What did the author do growing up?")
print(response)
```