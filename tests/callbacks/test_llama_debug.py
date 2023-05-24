"""Embeddings."""

from llama_index.callbacks.schema import CBEventType
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.llama_debug import LlamaDebugHandler


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
