# Usage Pattern

The usage pattern guide covers setup + usage of the `QueryPipeline` more in-depth.

Take a look at our [module usage guide](TODO) for more details on the supported components! 

## Setting up a Pipeline

Here we walk through a few different ways of setting up a query pipeline.

### Defining a Sequential Chain

Some simple pipelines are purely linear in nature - the output of the previous module directly goes into the input of the next module.

Some examples:
- prompt -> LLM -> output parsing
- prompt -> LLM -> prompt -> LLM
- retriever -> response synthesizer

These workflows can easily be expressed in the `QueryPipeline` through a simplified `chain` syntax.

### Defining a DAG




## Defining a Custom Query Component

You can easily define a custom component. Simply subclass a `QueryComponent`, implement validation/run functions + some helpers, and plug it in.


```python

from llama_index.query_pipeline import CustomQueryComponent
from typing import Dict, Any


class MyComponent(CustomQueryComponent):
    """My component."""
    
    # Pydantic class, put any attributes here
    ...

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
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

Generally you must make sure that for a link to work, the expected output and input types *roughly* line up.

We say roughly because we do some magic on existing modules to make sure that "stringable" outputs can be passed into
inputs that can be queried as a "string". Certain output types are treated as Stringable - `CompletionResponse`, `ChatResponse`, `Response`, `QueryBundle`, etc. Retrievers/query engines will automatically convert `string` inputs to `QueryBundle` objects.

This lets you do certain workflows that would otherwise require boilerplate string conversion if you were writing this yourself, for instance,
- LLM -> prompt, LLM -> retriever, LLM -> query engine
- query engine -> prompt, query engine -> retriever

If you are defining a custom component, you should use `_validate_component_inputs` to ensure that the inputs are the right type, and throw an error if they're not.

