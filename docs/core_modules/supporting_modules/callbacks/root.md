# Callbacks

## Concept
LlamaIndex provides callbacks to help debug, track, and trace the inner workings of the library. 
Using the callback manager, as many callbacks as needed can be added.

In addition to logging data related to events, you can also track the duration and number of occurances
of each event. 

Furthermore, a trace map of events is also recorded, and callbacks can use this data
however they want. For example, the `LlamaDebugHandler` will, by default, print the trace of events
after most operations.

**Callback Event Types**  
While each callback may not leverage each event type, the following events are available to be tracked:

- `CHUNKING` -> Logs for the before and after of text splitting.
- `NODE_PARSING` -> Logs for the documents and the nodes that they are parsed into.
- `EMBEDDING` -> Logs for the number of texts embedded.
- `LLM` -> Logs for the template and response of LLM calls.
- `QUERY` -> Keeps track of the start and end of each query.
- `RETRIEVE` -> Logs for the nodes retrieved for a query.
- `SYNTHESIZE` -> Logs for the result for synthesize calls.
- `TREE` -> Logs for the summary and level of summaries generated.
- `SUB_QUESTION` -> Log for a generated sub question and answer.

You can implement your own callback to track and trace these events, or use an existing callback.


## Modules

Currently supported callbacks are as follows:

- [TokenCountingHandler](/examples/callbacks/TokenCountingHandler.ipynb) -> Flexible token counting for prompt, completion, and embedding token usage. See the migration details [here](/core_modules/model_modules/callbacks/token_counting_migration.md)
- [LlamaDebugHanlder](/examples/callbacks/LlamaDebugHandler.ipynb) -> Basic tracking and tracing for events. Example usage can be found in the notebook below.
- [WandbCallbackHandler](/examples/callbacks/WandbCallbackHandler.ipynb) -> Tracking of events and traces using the Wandb Prompts frontend. More details are in the notebook below or at [Wandb](https://docs.wandb.ai/guides/prompts/quickstart)
- [AimCallback](/examples/callbacks/AimCallback.ipynb) -> Tracking of LLM inputs and outputs. Example usage can be found in the notebook below.
- [OpenInferenceCallbackHandler](/docs/examples/callbacks/OpenInferenceCallback.ipynb) -> Tracking of AI model inferences. Example usage can be found in the notebook below.
- [OpenAIFineTuningHandler](https://github.com/jerryjliu/llama_index/blob/main/experimental/openai_fine_tuning/openai_fine_tuning.ipynb) -> Records all LLM inputs and outputs. Then, provides a function `save_finetuning_events()` to save inputs and outputs in a format suitable for fine-tuning with OpenAI.


```{toctree}
---
maxdepth: 1
hidden:
---
/examples/callbacks/TokenCountingHandler.ipynb
/examples/callbacks/LlamaDebugHandler.ipynb
/examples/callbacks/WandbCallbackHandler.ipynb
/examples/callbacks/AimCallback.ipynb
/examples/callbacks/OpenInferenceCallback.ipynb
token_counting_migration.md
```
