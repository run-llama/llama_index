"""Embeddings."""

from gpt_index.callbacks.schema import CBEventType
from gpt_index.callbacks.llama_debug import LlamaDebugHandler

TEST_PAYLOAD = {"one": 1, "two": 2}
TEST_ID = "my id"


def test_on_event_start() -> None:
    """Test event start."""
    handler = LlamaDebugHandler()

    event_id = handler.on_event_start(
        CBEventType.LLM, payload=TEST_PAYLOAD, event_id=TEST_ID
    )

    assert event_id == TEST_ID
    assert len(handler.events) == 1
    assert len(handler.sequential_events) == 1

    events = handler.events.get(CBEventType.LLM)
    assert isinstance(events, list)
    assert events[0].payload == TEST_PAYLOAD


def test_on_event_end() -> None:
    """Test event end."""
    handler = LlamaDebugHandler()

    handler.on_event_end(CBEventType.EMBEDDING, payload=TEST_PAYLOAD, event_id=TEST_ID)

    assert len(handler.events) == 1
    assert len(handler.sequential_events) == 1

    events = handler.events.get(CBEventType.EMBEDDING)
    assert isinstance(events, list)
    assert events[0].payload == TEST_PAYLOAD
    assert events[0].id == TEST_ID


def test_get_event_stats() -> None:
    """Test get event stats."""
    handler = LlamaDebugHandler()

    event_id = handler.on_event_start(CBEventType.CHUNKING, payload=TEST_PAYLOAD)
    handler.on_event_end(CBEventType.CHUNKING, event_id=event_id)

    assert len(handler.events[CBEventType.CHUNKING]) == 2

    event_stats = handler.get_event_time_info(CBEventType.CHUNKING)

    assert event_stats.total_count == 1
    assert event_stats.total_secs == 0


def test_flush_events() -> None:
    """Test flush events."""
    handler = LlamaDebugHandler()

    event_id = handler.on_event_start(CBEventType.CHUNKING, payload=TEST_PAYLOAD)
    handler.on_event_end(CBEventType.CHUNKING, event_id=event_id)

    event_id = handler.on_event_start(CBEventType.CHUNKING, payload=TEST_PAYLOAD)
    handler.on_event_end(CBEventType.CHUNKING, event_id=event_id)

    assert len(handler.events[CBEventType.CHUNKING]) == 4

    handler.flush_event_logs()

    assert len(handler.events) == 0
    assert len(handler.sequential_events) == 0
