from contextvars import ContextVar
from typing import Any, Dict, List

from llama_index.core.types import Thread


def test_thread_with_no_target() -> None:
    """Test that creating a Thread with target=None does not crash."""
    # Should not raise exception
    t = Thread(target=None)
    t.start()
    t.join()


def test_thread_with_target() -> None:
    """Test that a Thread with a target runs the target."""
    result: List[int] = []

    def target_fn() -> None:
        result.append(1)

    t = Thread(target=target_fn)
    t.start()
    t.join()
    assert result == [1]


def test_thread_context_copy() -> None:
    """Test that the context is copied to the new thread."""
    var = ContextVar("var", default=0)
    var.set(1)

    results: Dict[str, Any] = {}

    def target_fn() -> None:
        results["value"] = var.get()

    t = Thread(target=target_fn)
    t.start()
    t.join()

    # It should copy the context where var=1.
    # If it didn't use copy_context(), it might still work in some threaded envs
    # depending on how context vars propagate, but Thread implementation explicitly uses copy_context().run
    assert results["value"] == 1
