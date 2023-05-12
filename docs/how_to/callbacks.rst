📞 Callbacks
==============================

LlamaIndex provides callbacks to help debug, track, and trace the inner workings of the library. 
Using the callback manager, as many callbacks as needed can be added.

In addition to logging data related to events, you can also track the duration and number of occurances
of each event.

While each callback may not leverage each event type, the following events are available to be tracked:

- CHUNKING -> Logs for the before and after of text splitting.
- NODE_PARSING -> Logs for the documents and the nodes that they are parsed into.
- EMBEDDING -> Logs for the number of texts embedded.
- LLM -> Logs for the template and response of LLM calls.
- QUERY -> Keeps track of the start and end of each query.
- RETRIEVE -> Logs for the nodes retrieved for a query.
- SYNTHESIZE -> Logs for the result for synthesize calls.
- TREE -> Logs for the summary and level of summaries generated.

You can implement your own callback to track these events, or use an existing callback.

Complete examples can be found in the notebooks below:

- [LlamaDebugHandler](../examples/callbacks/LlamaDebugHandler.ipynb)
- [AimCallback](../examples/callbacks/AimCallback.ipynb)

And the API reference can be found [here](../../reference/callbacks.rst).

.. toctree::
   :maxdepth: 1
   :caption: Callbacks

   ../examples/callbacks/LlamaDebugHandler.ipynb
   ../examples/callbacks/AimCallback.ipynb
   ../../reference/callbacks.rst