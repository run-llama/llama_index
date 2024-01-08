# Usage Pattern

The usage pattern guide covers setup + usage of the `QueryPipeline` more in-depth.

## Setting up a Pipeline

Here we walk through a few different ways of setting up a query pipeline.

### Defining a Sequential Chain

Some simple pipelines are purely linear in nature - the output of the previous module directly goes into the input of the next module.

Some examples:

- prompt -> LLM -> output parsing
- prompt -> LLM -> prompt -> LLM
- retriever -> response synthesizer

These workflows can easily be expressed in the `QueryPipeline` through a simplified `chain` syntax.

```python
from llama_index.query_pipeline.query import QueryPipeline

# try chaining basic prompts
prompt_str = "Please generate related movies to {movie_name}"
prompt_tmpl = PromptTemplate(prompt_str)
llm = OpenAI(model="gpt-3.5-turbo")

p = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)
```

### Defining a DAG

Many pipelines will require you to setup a DAG (for instance, if you want to implement all the steps in a standard RAG pipeline).

Here we offer a lower-level API to add modules along with their keys, and define links between previous module outputs to next
module inputs.

```python
from llama_index.postprocessor import CohereRerank
from llama_index.response_synthesizers import TreeSummarize
from llama_index import ServiceContext

# define modules
prompt_str = "Please generate a question about Paul Graham's life regarding the following topic {topic}"
prompt_tmpl = PromptTemplate(prompt_str)
llm = OpenAI(model="gpt-3.5-turbo")
retriever = index.as_retriever(similarity_top_k=3)
reranker = CohereRerank()
summarizer = TreeSummarize(
    service_context=ServiceContext.from_defaults(llm=llm)
)

# define query pipeline
p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "llm": llm,
        "prompt_tmpl": prompt_tmpl,
        "retriever": retriever,
        "summarizer": summarizer,
        "reranker": reranker,
    }
)
p.add_link("prompt_tmpl", "llm")
p.add_link("llm", "retriever")
p.add_link("retriever", "reranker", dest_key="nodes")
p.add_link("llm", "reranker", dest_key="query_str")
p.add_link("reranker", "summarizer", dest_key="nodes")
p.add_link("llm", "summarizer", dest_key="query_str")
```

## Running the Pipeline

### Single-Input/Single-Output

The input is the kwargs of the first component.

If the output of the last component is a single object (and not a dictionary of objects), then we return that directly.

Taking the pipeline in the previous example, the output will be a `Response` object since the last step is the `TreeSummarize` response synthesis module.

```python
output = p.run(topic="YC")
# output type is Response
type(output)
```

### Multi-Input/Multi-Output

If your DAG has multiple root nodes / and-or output nodes, you can try `run_multi`. Pass in an input dictionary containing module key -> input dict. Output is dictionary of module key -> output dict.

If we ran the prev example,

```python
output_dict = p.run_multi({"llm": {"topic": "YC"}})
print(output_dict)

# output dict is {"summarizer": {"output": response}}
```

### Defining partials

If you wish to prefill certain inputs for a module, you can do so with `partial`! Then the DAG would just hook into the unfilled inputs.

You may need to convert a module via `as_query_component`.

Here's an example:

```python
summarizer = TreeSummarize(
    service_context=ServiceContext.from_defaults(llm=llm)
)
summarizer_c = summarizer.as_query_component(partial={"nodes": nodes})
# can define a chain because llm output goes into query_str, nodes is pre-filled
p = QueryPipeline(chain=[prompt_tmpl, llm, summarizer_c])
# run pipeline
p.run(topic="YC")
```

(query-pipeline-custom-component)=

## Defining a Custom Query Component

You can easily define a custom component. Simply subclass a `QueryComponent`, implement validation/run functions + some helpers, and plug it in.

```python
from llama_index.query_pipeline import CustomQueryComponent
from typing import Dict, Any


class MyComponent(CustomQueryComponent):
    """My component."""

    # Pydantic class, put any attributes here
    ...

    def _validate_component_inputs(
        self, input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        # NOTE: this is OPTIONAL but we show you here how to do validation as an example
        return input

    @property
    def _input_keys(self) -> set:
        """Input keys dict."""
        return {"input_key1", ...}

    @property
    def _output_keys(self) -> set:
        # can do multi-outputs too
        return {"output_key"}

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        """Run the component."""
        # run logic
        ...
        return {"output_key": result}
```

For more details check out our [in-depth query transformations guide](/examples/pipeline/query_pipeline.ipynb).

## Ensuring outputs are compatible

By linking modules within a `QueryPipeline`, the output of one module goes into the input of the next module.

Generally you must make sure that for a link to work, the expected output and input types _roughly_ line up.

We say roughly because we do some magic on existing modules to make sure that "stringable" outputs can be passed into
inputs that can be queried as a "string". Certain output types are treated as Stringable - `CompletionResponse`, `ChatResponse`, `Response`, `QueryBundle`, etc. Retrievers/query engines will automatically convert `string` inputs to `QueryBundle` objects.

This lets you do certain workflows that would otherwise require boilerplate string conversion if you were writing this yourself, for instance,

- LLM -> prompt, LLM -> retriever, LLM -> query engine
- query engine -> prompt, query engine -> retriever

If you are defining a custom component, you should use `_validate_component_inputs` to ensure that the inputs are the right type, and throw an error if they're not.
