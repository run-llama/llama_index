class Dispatcher:
    def __init__(
        self,
        handlers,
    ) -> None:
        self.handlers = handlers

    def add_handler(self, handler) -> None:
        """Add handler to set of handlers."""
        self.handlers += [handler]

    def dispatch(self, event) -> None:
        """Dispatch event to all registered handlers."""
        for h in self.handlers:
            h.handle(event)
