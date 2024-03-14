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

.. autopydantic_model:: llama_index.core.instrumentation.events.agent

.. autopydantic_model:: llama_index.core.instrumentation.events.chat_engine

.. autopydantic_model:: llama_index.core.instrumentation.events.embedding

.. autopydantic_model:: llama_index.core.instrumentation.events.llm

.. autopydantic_model:: llama_index.core.instrumentation.events.query

.. autopydantic_model:: llama_index.core.instrumentation.events.retrieval

.. autopydantic_model:: llama_index.core.instrumentation.events.synthesis

Span Handlers
-------------

.. autopydantic_model:: llama_index.core.instrumentation.span_handlers.base.BaseSpanHandler

.. autopydantic_model:: llama_index.core.instrumentation.span_handlers.simple.SimpleSpanHandler

Spans Types
-----------

.. autopydantic_model:: llama_index.core.instrumentation.span.base.BaseSpan

.. autopydantic_model:: llama_index.core.instrumentation.span.simple.SimpleSpan
