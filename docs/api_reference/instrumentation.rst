Instrumentation
===============

LlamaIndex contains a simple instrumentation framework that allows you to
observe events and spans happening in the framework.

Event Handlers
--------------

.. autopydantic_model:: llama_index.core.instrumentation.event_handlers.base.BaseEventHandler

Event Types
-----------

.. autopydantic_model:: llama_index.core.instrumentation.events.base.BaseEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.agent.AgentChatWithStepEndEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.agent.AgentChatWithStepStartEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.agent.AgentRunStepEndEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.agent.AgentRunStepStartEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.agent.AgentToolCallEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.chat_engine.StreamChatDeltaReceivedEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.chat_engine.StreamChatEndEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.chat_engine.StreamChatErrorEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.chat_engine.StreamChatStartEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.embedding.EmbeddingEndEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.embedding.EmbeddingStartEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.llm.LLMChatEndEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.llm.LLMChatStartEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.llm.LLMCompletionEndEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.llm.LLMCompletionStartEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.llm.LLMPredictEndEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.llm.LLMPredictStartEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.query.QueryEndEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.query.QueryStartEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.retrieval.RetrievalEndEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.retrieval.RetrievalStartEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.synthesis.GetResponseEndEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.synthesis.GetResponseStartEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.synthesis.SynthesizeEndEvent

.. autopydantic_model:: llama_index.core.instrumentation.events.synthesis.SynthesizeStartEvent

Span Handlers
-------------

.. autopydantic_model:: llama_index.core.instrumentation.span_handlers.base.BaseSpanHandler

.. autopydantic_model:: llama_index.core.instrumentation.span_handlers.simple.SimpleSpanHandler

Spans Types
-----------

.. autopydantic_model:: llama_index.core.instrumentation.span.base.BaseSpan
