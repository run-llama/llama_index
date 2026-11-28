from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.settings import Settings


class MockCallbackHandler(BaseCallbackHandler):
    def __init__(self, name: str = "mock") -> None:
        super().__init__([], [])
        self.name = name

    def on_event_start(self, *args, **kwargs):
        pass

    def on_event_end(self, *args, **kwargs):
        pass

    def start_trace(self, *args, **kwargs):
        pass

    def end_trace(self, *args, **kwargs):
        pass


class GlobalMockHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__([], [])

    def on_event_start(self, *args, **kwargs):
        pass

    def on_event_end(self, *args, **kwargs):
        pass

    def start_trace(self, *args, **kwargs):
        pass

    def end_trace(self, *args, **kwargs):
        pass


def test_callback_manager_does_not_mutate_caller_handlers(monkeypatch) -> None:
    # Set a global handler to trigger appending in __init__
    import llama_index.core

    monkeypatch.setattr(llama_index.core, "global_handler", GlobalMockHandler())

    handler1 = MockCallbackHandler("h1")
    caller_handlers = [handler1]

    cm1 = CallbackManager(handlers=caller_handlers)

    # Caller's original list should not be modified
    assert len(caller_handlers) == 1
    assert caller_handlers == [handler1]
    assert len(cm1.handlers) == 2

    # Reusing caller_handlers in another CallbackManager should succeed
    cm2 = CallbackManager(handlers=caller_handlers)
    assert len(caller_handlers) == 1
    assert len(cm2.handlers) == 2


def test_callback_manager_add_handler_does_not_mutate_caller_handlers() -> None:
    handler1 = MockCallbackHandler("h1")
    caller_handlers = [handler1]

    cm = CallbackManager(handlers=caller_handlers)
    cm.add_handler(MockCallbackHandler("h2"))

    assert len(caller_handlers) == 1
    assert caller_handlers == [handler1]
    assert len(cm.handlers) == 2


def test_callback_manager_set_handlers_does_not_mutate_caller_handlers() -> None:
    handler1 = MockCallbackHandler("h1")
    caller_handlers = [handler1]

    cm = CallbackManager()
    cm.set_handlers(caller_handlers)
    cm.add_handler(MockCallbackHandler("h2"))

    assert len(caller_handlers) == 1
    assert caller_handlers == [handler1]
    assert len(cm.handlers) == 2


def test_callback_manager_copies_settings_default_handlers(monkeypatch) -> None:
    base_handler = MockCallbackHandler("base")
    default_cm = CallbackManager(handlers=[base_handler])
    monkeypatch.setattr(Settings, "_callback_manager", default_cm)

    cm = CallbackManager()
    cm.add_handler(MockCallbackHandler("extra"))

    assert len(default_cm.handlers) == 1
    assert default_cm.handlers == [base_handler]
    assert len(cm.handlers) == 2
