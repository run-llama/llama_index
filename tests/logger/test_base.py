"""Unit tests for logger."""

from llama_index.logger.base import LlamaLogger


def test_logger() -> None:
    """Test logger."""
    logger = LlamaLogger()
    # test add
    for i in range(4):
        logger.add_log({"foo": "bar", "item": i})
    logs = logger.get_logs()
    assert logs == [
        {"foo": "bar", "item": 0},
        {"foo": "bar", "item": 1},
        {"foo": "bar", "item": 2},
        {"foo": "bar", "item": 3},
    ]

    # test reset
    logger.reset()
    assert logger.get_logs() == []


def test_logger_metadata() -> None:
    """Test logger metadata."""
    logger = LlamaLogger()
    # first add
    for i in range(2):
        logger.add_log({"foo": "bar", "item": i})
    # set metadata
    logger.set_metadata({"baz": "qux"})

    for i in range(2, 4):
        logger.add_log({"foo": "bar", "item": i})

    logger.unset_metadata({"baz"})

    for i in range(4, 6):
        logger.add_log({"foo": "bar", "item": i})

    logs = logger.get_logs()

    assert logs == [
        {"foo": "bar", "item": 0},
        {"foo": "bar", "item": 1},
        {"foo": "bar", "item": 2, "baz": "qux"},
        {"foo": "bar", "item": 3, "baz": "qux"},
        {"foo": "bar", "item": 4},
        {"foo": "bar", "item": 5},
    ]
