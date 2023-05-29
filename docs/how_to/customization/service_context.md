# ServiceContext

The ServiceContext object encapsulates the resources used to create indexes and run queries.

The following optional items can be set in the service context:

- llm_predictor: The LLM used to generate natural language responses to queries.
- embed_model: The embedding model used to generate vector representations of text.
- prompt_helper: The PromptHelper object that defines settings for text sent to the LLM.
- node_parser: The parser that converts documents into nodes.
- chunk_size_limit: The maximum size of a node. Is used for the prompt helper and node parser when they aren't provided.
- callback_managaer: The callback manager object that calls it's handlers on events. Provides basic logging and tracing capabilities.

Here's a complete example that sets up all objects using their default settings:

```python
from langchain.llms import OpenAI
from llama_index import ServiceContext, LLMPredictor, OpenAIEmbedding, PromptHelper
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser

llm_predictor = LLMPredictor(llm=OpenAI(model_name='text-davinci-003', temperature=0))
embed_model = OpenAIEmbedding()
node_parser = SimpleNodeParser(text_splitter=TokenTextSplitter(chunk_size=1024, chunk_overlap=200))
prompt_helper = PromptHelper(max_input_size=4096, num_output=256, max_chunk_overlap=20, chunk_size_limit=1024)
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

Here's a quick example of what a global service context might look like. This service context changes the LLM to `gpt-3.5-turbo`, changes the `chunk_size_limit`, and sets up a `callback_manager` to trace events using the `LlamaDebugHandler`.

First, define the service context:

```python
from langchain.chat_models import ChatOpenAI
from llama_index import ServiceContext, LLMPredictor
from llama_index.callbacks import CallbackManager, LlamaDebugHandler

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
llm_predictor = LLMPredictor(llm=llm)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512, callback_manager=callback_manager, is_global=True)
```

Then, set the global service context object

```python
from llama_index import set_global_service_context
set_global_service_context(service_context)
```
