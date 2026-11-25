"""Embeddings."""

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.llama_debug import LlamaDebugHandler
from llama_index.core.callbacks.schema import CBEventType

TEST_PAYLOAD = {"one": 1, "two": 2}
TEST_ID = "my id"


def test_on_event_start() -> None:
    """Test event start."""
    handler = LlamaDebugHandler()

    event_id = handler.on_event_start(
        CBEventType.LLM, payload=TEST_PAYLOAD, event_id=TEST_ID
    )

    assert event_id == TEST_ID
    assert len(handler.event_pairs_by_type) == 1
    assert len(handler.sequential_events) == 1

    events = handler.event_pairs_by_type.get(CBEventType.LLM)
    assert isinstance(events, list)
    assert events[0].payload == TEST_PAYLOAD


def test_on_event_end() -> None:
    """Test event end."""
    handler = LlamaDebugHandler()

    handler.on_event_end(CBEventType.EMBEDDING, payload=TEST_PAYLOAD, event_id=TEST_ID)

    assert len(handler.event_pairs_by_type) == 1
    assert len(handler.sequential_events) == 1

    events = handler.event_pairs_by_type.get(CBEventType.EMBEDDING)
    assert isinstance(events, list)
    assert events[0].payload == TEST_PAYLOAD
    assert events[0].id_ == TEST_ID


def test_get_event_stats() -> None:
    """Test get event stats."""
    handler = LlamaDebugHandler()

    event_id = handler.on_event_start(CBEventType.CHUNKING, payload=TEST_PAYLOAD)
    handler.on_event_end(CBEventType.CHUNKING, event_id=event_id)

    assert len(handler.event_pairs_by_type[CBEventType.CHUNKING]) == 2

    event_stats = handler.get_event_time_info(CBEventType.CHUNKING)

    assert event_stats.total_count == 1
    assert event_stats.total_secs > 0.0


def test_flush_events() -> None:
    """Test flush events."""
    handler = LlamaDebugHandler()

    event_id = handler.on_event_start(CBEventType.CHUNKING, payload=TEST_PAYLOAD)
    handler.on_event_end(CBEventType.CHUNKING, event_id=event_id)

    event_id = handler.on_event_start(CBEventType.CHUNKING, payload=TEST_PAYLOAD)
    handler.on_event_end(CBEventType.CHUNKING, event_id=event_id)

    assert len(handler.event_pairs_by_type[CBEventType.CHUNKING]) == 4

    handler.flush_event_logs()

    assert len(handler.event_pairs_by_type) == 0
    assert len(handler.sequential_events) == 0


def test_ignore_events() -> None:
    """Test ignore event starts and ends."""
    handler = LlamaDebugHandler(
        event_starts_to_ignore=[CBEventType.CHUNKING],
        event_ends_to_ignore=[CBEventType.LLM],
    )
    manager = CallbackManager([handler])

    event_id = manager.on_event_start(CBEventType.CHUNKING, payload=TEST_PAYLOAD)
    manager.on_event_end(CBEventType.CHUNKING, event_id=event_id)

    event_id = manager.on_event_start(CBEventType.LLM, payload=TEST_PAYLOAD)
    manager.on_event_end(CBEventType.LLM, event_id=event_id)

    event_id = manager.on_event_start(CBEventType.EMBEDDING, payload=TEST_PAYLOAD)
    manager.on_event_end(CBEventType.EMBEDDING, event_id=event_id)

    # should have only captured 6 - 2 = 4 events
    assert len(handler.sequential_events) == 4


def test_end_trace_plain_dict_and_no_defaultdict_pollution() -> None:
    """Test end_trace with a plain dict trace_map does not raise KeyError or pollute events_pairs_by_id."""
    handler = LlamaDebugHandler(print_trace_on_end=True)
    handler.start_trace("query")
    event_id = handler.on_event_start(
        CBEventType.LLM, payload=TEST_PAYLOAD, event_id=TEST_ID
    )
    handler.on_event_end(CBEventType.LLM, payload=TEST_PAYLOAD, event_id=event_id)

    # Passing a plain dict (where leaf event_id is not a key) must not raise KeyError,
    # and printing the trace map must not insert 'root': [] into events_pairs_by_id.
    handler.end_trace(trace_id="query", trace_map={"root": [event_id]})
    assert "root" not in handler.events_pairs_by_id
    assert list(handler.events_pairs_by_id.keys()) == [TEST_ID]

    stats = handler._get_time_stats_from_event_pairs(
        list(handler.events_pairs_by_id.values())
    )
    assert stats.total_count == 1


def test_read_queries_do_not_mutate_event_pairs_by_type_and_flush_resets_trace() -> None:
    """Test read-only queries do not insert empty keys and flush_event_logs resets trace state."""
    handler = LlamaDebugHandler(print_trace_on_end=False)
    assert handler.get_events(CBEventType.LLM) == []
    assert handler.get_event_pairs(CBEventType.EMBEDDING) == []
    assert handler.get_llm_inputs_outputs() == []
    assert len(handler.event_pairs_by_type) == 0

    handler.start_trace("trace-1")
    handler.end_trace("trace-1", trace_map={"root": ["e1"]})
    handler.flush_event_logs()
    assert handler._cur_trace_id is None
    assert len(handler._trace_map) == 0

