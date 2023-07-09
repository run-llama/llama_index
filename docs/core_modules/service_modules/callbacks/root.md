# Callbacks

## Concept
LlamaIndex provides callbacks to help debug, track, and trace the inner workings of the library. 
Using the callback manager, as many callbacks as needed can be added.

In addition to logging data related to events, you can also track the duration and number of occurances
of each event. 

Furthermore, a trace map of events is also recorded, and callbacks can use this data
however they want. For example, the :code:`LlamaDebugHandler` will, by default, print the trace of events
after most operations.

While each callback may not leverage each event type, the following events are available to be tracked:

- CHUNKING -> Logs for the before and after of text splitting.
- NODE_PARSING -> Logs for the documents and the nodes that they are parsed into.
- EMBEDDING -> Logs for the number of texts embedded.
- LLM -> Logs for the template and response of LLM calls.
- QUERY -> Keeps track of the start and end of each query.
- RETRIEVE -> Logs for the nodes retrieved for a query.
- SYNTHESIZE -> Logs for the result for synthesize calls.
- TREE -> Logs for the summary and level of summaries generated.
- SUB_QUESTIONS -> Logs for the sub questions and answers generated.

You can implement your own callback to track and trace these events, or use an existing callback.

Currently supported callbacks are as follows:

- TokenCountingHandler -> Flexible token counting for prompt, completion, and embedding token usage. `See the migration details here <./callbacks/token_counting_migration.html>`_ 
- LlamaDebugHanlder -> Basic tracking and tracing for events. Example usage can be found in the notebook below.
- WandbCallbackHandler -> Tracking of events and traces using the Wandb Prompts frontend. More details are in the notebook below or at `Wandb <https://docs.wandb.ai/guides/prompts/quickstart>`_
- AimCallback -> Tracking of LLM inputs and outputs. Example usage can be found in the notebook below.


## Modules

```{toctree}
---
maxdepth: 1
---
/examples/callbacks/TokenCountingHandler.ipynb
/examples/callbacks/LlamaDebugHandler.ipynb
/examples/callbacks/WandbCallbackHandler.ipynb
/examples/callbacks/AimCallback.ipynb
./callbacks/token_counting_migration.md
