from collections import defaultdict
import asyncio
from typing import Dict, Any, Optional, List, Type, TYPE_CHECKING

from .events import Event

if TYPE_CHECKING:
    from .session import WorkflowSession


class Context:
    """A global object representing a context for a given workflow run.

    The Context object can be used to store data that needs to be available across iterations during a workflow
    execution, and across multiple workflow runs.
    Every context instance offers two type of data storage: a global one, that's shared among all the steps within a
    workflow, and private one, that's only accessible from a single step.

    Both `set` and `get` operations on global data are governed by a lock, and considered coroutine-safe.
    """

    def __init__(
        self,
        session: Optional["WorkflowSession"] = None,
        parent: Optional["Context"] = None,
    ) -> None:
        # Global data storage
        if parent is not None:
            self._globals = parent._globals
        else:
            self._globals: Dict[str, Any] = {}
            self._lock = asyncio.Lock()
            if session is None:
                msg = "A workflow session is needed to create a root context"
                raise ValueError(msg)
            self._session = session

        # Local data storage
        self._locals: Dict[str, Any] = {}

        # Step-specific instance
        self._parent: Optional[Context] = parent
        self._events_buffer: Dict[Type[Event], List[Event]] = defaultdict(list)

    async def set(self, key: str, value: Any, make_private: bool = False) -> None:
        """Store `value` into the Context under `key`.

        Args:
            key: A unique string to identify the value stored.
            value: The data to be stored.
            make_private: Make the value only accessible from the step that stored it.

        Raises:
            ValueError: When make_private is True but a key already exists in the global storage.
        """
        if make_private:
            if key in self._globals:
                msg = f"A key named '{key}' already exists in the Context storage."
                raise ValueError(msg)
            self._locals[key] = value
            return

        async with self.lock:
            self._globals[key] = value

    async def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get the value corresponding to `key` from the Context.

        Args:
            key: A unique string to identify the value stored.
            default: The value to return when `key` is missing instead of raising an exception.

        Raises:
            ValueError: When there's not value accessible corresponding to `key`.
        """
        if key in self._locals:
            return self._locals[key]
        elif key in self._globals:
            async with self.lock:
                return self._globals[key]
        elif default is not None:
            return default

        msg = f"Key '{key}' not found in Context"
        raise ValueError(msg)

    @property
    def data(self):
        """This property is provided for backward compatibility.

        Use `get` and `set` instead.
        """
        return self._globals

    @property
    def lock(self) -> asyncio.Lock:
        """Returns a mutex to lock the Context."""
        return self._parent._lock if self._parent else self._lock

    @property
    def session(self) -> "WorkflowSession":
        """Returns a mutex to lock the Context."""
        return self._parent._session if self._parent else self._session

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
