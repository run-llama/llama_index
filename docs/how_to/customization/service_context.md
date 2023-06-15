# ServiceContext

The ServiceContext object encapsulates the resources used to create indexes and run queries.

The following optional items can be set in the service context:

- llm_predictor: The LLM used to generate natural language responses to queries.
- embed_model: The embedding model used to generate vector representations of text.
- prompt_helper: The PromptHelper object that helps with truncating and repacking text chunks to fit in the LLM's context window.
- node_parser: The parser that converts documents into nodes.
- callback_managaer: The callback manager object that calls it's handlers on events. Provides basic logging and tracing capabilities.

We also expose some common kwargs (of the above components) via the `ServiceContext.from_defaults` method
for convenience (so you don't have to manually construct them).
 
Kwargs node parser:
- chunk_size: The size of the text chunk for a node . Is used for the node parser when they aren't provided.
- chunk overlap: The amount of overlap between nodes.

Kwargs for prompt helper:
- context_window: The size of the context window of the LLM. Typically we set this 
  automatically with the model metadata. But we also allow explicit override via this parameter
  for additional control (or in case the default is not available for certain latest
  models)
- num_output: The number of maximum output from the LLM. Typically we set this
  automatically given the model metadata. This parameter does not actually limit the model
  output, it affects the amount of "space" we save for the output, when computing 
  available context window size for packing text from retrieved Nodes.

Here's a complete example that sets up all objects using their default settings:

```python
from langchain.llms import OpenAI
from llama_index import ServiceContext, LLMPredictor, OpenAIEmbedding, PromptHelper
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser

llm_predictor = LLMPredictor(llm=OpenAI(model_name='text-davinci-003', temperature=0, max_tokens=256))
embed_model = OpenAIEmbedding()
node_parser = SimpleNodeParser(text_splitter=TokenTextSplitter(chunk_size=1024, chunk_overlap=20))
prompt_helper = PromptHelper(context_window=4096, num_output=256, chunk_overlap_ratio=0.1, chunk_size_limit=None)
service_context = ServiceContext.from_defaults(
  llm_predictor=llm_predictor,
  embed_model=embed_model,
  node_parser=node_parser,
  prompt_helper=prompt_helper
)
```

## Global ServiceContext

You can specify a different set of defaults for the ServiceContext by setting up a global service context.

With a global service context, any attributes not provided when calling `ServiceContext.from_defaults()` will be pulled from your global service context. If you never define a service context anywhere else, then the global service context will always be used.

Here's a quick example of what a global service context might look like. This service context changes the LLM to `gpt-3.5-turbo`, changes the `chunk_size`, and sets up a `callback_manager` to trace events using the `LlamaDebugHandler`.

First, define the service context:

```python
from langchain.chat_models import ChatOpenAI
from llama_index import ServiceContext, LLMPredictor
from llama_index.callbacks import CallbackManager, LlamaDebugHandler

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
llm_predictor = LLMPredictor(llm=llm)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size=512, callback_manager=callback_manager)
```

Then, set the global service context object

```python
from llama_index import set_global_service_context
set_global_service_context(service_context)
```
