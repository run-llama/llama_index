# Module Usage

Currently the following LlamaIndex modules are supported within a QueryPipeline. Remember, you can define your own!

### LLMs (both completion and chat)

- Base class: `LLM`
- [Module Guide](/module_guides/models/llms.md)
- If chat model:
  - Input: `messages`. Takes in any `List[ChatMessage]` or any stringable input.
  - Output: `output`. Outputs `ChatResponse` (stringable)
- If completion model:
  - Input: `prompt`. Takes in any stringable input.
  - Output: `output`. Outputs `CompletionResponse` (stringable)

### Prompts

- Base class: `PromptTemplate`
- [Module Guide](/module_guides/models/prompts.md)
- Input: Prompt template variables. Each variable can be a stringable input.
- Output: `output`. Outputs formatted prompt string (stringable)

### Query Engines

- Base class: `BaseQueryEngine`
- [Module Guide](/module_guides/deploying/query_engine/root.md)
- Input: `input`. Takes in any stringable input.
- Output: `output`. Outputs `Response` (stringable)

### Query Transforms

- Base class: `BaseQueryTransform`
- [Module Guide](/optimizing/advanced_retrieval/query_transformations.md)
- Input: `query_str`, `metadata` (optional). `query_str` is any stringable input.
- Output: `query_str`. Outputs string.

### Retrievers

- Base class: `BaseRetriever`
- [Module Guide](/module_guides/querying/retriever/root.md)
- Input: `input`. Takes in any stringable input.
- Output: `output`. Outputs list of nodes `List[BaseNode]`.

### Output Parsers

- Base class: `BaseOutputParser`
- [Module Guide](/module_guides/querying/structured_outputs/output_parser.md)
- Input: `input`. Takes in any stringable input.
- Output: `output`. Outputs whatever type output parser is supposed to parse out.

### Postprocessors/Rerankers

- Base class: `BaseNodePostprocessor`
- [Module Guide](/module_guides/querying/node_postprocessors/root.md)
- Input: `nodes`, `query_str` (optional). `nodes` is `List[BaseNode]`, `query_str` is any stringable input.
- Output: `nodes`. Outputs list of nodes `List[BaseNode]`.

### Response Synthesizers

- Base class: `BaseSynthesizer`
- [Module Guide]()
- Input: `nodes`, `query_str`. `nodes` is `List[BaseNode]`, `query_str` is any stringable input.
- Output: `output`. Outputs `Response` object (stringable).

### Other QueryPipeline objects

You can define a `QueryPipeline` as a module within another query pipeline. This makes it easy for you to string together complex workflows.

### Custom Components

See our [custom components guide](query-pipeline-custom-component) for more details.
