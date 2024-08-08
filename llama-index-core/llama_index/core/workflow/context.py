from collections import defaultdict
import asyncio
from typing import Dict, Any, Optional, List, Type
from types import TracebackType

from .events import Event


class Context:
    """A global object representing a context for a given workflow run.

    The Context object can be used to store data that needs to be shared across iterations during a workflow execution.
    Steps can use data stored in a Context object to keep a state across multiple executions within a workflow run.
    Any Context instance offers two type of data storage: `ctx.globals`, that's shared among all the steps within a
    workflow, and `ctx.locals`, that's private to a single step.

    Note that using `ctx.globals` is not coroutine-safe if you `await` for something in-between accessing
    the global state. In that case, make sure to lock the Context object before accessing `ctx.globals`. You can also
    use the Context object as an async context manager:

        with ctx:
            ctx.globals["foo"] = "bar"
            await some_async_call()
            ctx.globals["foo"] = "baz"
    """

    def __init__(self, parent: Optional["Context"] = None) -> None:
        # Global data storage
        if parent:
            self._globals = parent.data
        else:
            self._globals: Dict[str, Any] = {}
            self._lock = asyncio.Lock()

        # Local data storage
        self._locals: Dict[str, Any] = {}

        # Step-specific instance
        self._parent: Optional[Context] = parent
        self._events_buffer: Dict[Type[Event], List[Event]] = defaultdict(list)

    @property
    def globals(self) -> Dict[str, Any]:
        """Returns the local storage."""
        return self._globals

    @property
    def data(self):
        """This property is provided for backward compatibility.

        Use `globals` instead.
        """
        return self._globals

    @property
    def locals(self) -> Dict[str, Any]:
        """Returns the local storage."""
        return self._locals

    @property
    def lock(self) -> asyncio.Lock:
        """Returns a mutex to lock the Context."""
        return self._parent._lock if self._parent else self._lock

    async def __aenter__(self) -> "Context":
        await self.lock.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        self.lock.release()
        return None

    def collect_events(
        self, ev: Event, expected: List[Type[Event]]
    ) -> Optional[List[Event]]:
        self._events_buffer[type(ev)].append(ev)

        retval: List[Event] = []
        for e_type in expected:
            e_instance_list = self._events_buffer.get(e_type)
            if e_instance_list:
                retval.append(e_instance_list.pop(0))

        if len(retval) == len(expected):
            return retval

        # put back the events if unable to collect all
        for ev in retval:
            self._events_buffer[type(ev)].append(ev)

        return None
