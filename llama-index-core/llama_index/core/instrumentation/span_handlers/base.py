import inspect
import threading
from abc import abstractmethod
from typing import Any, Dict, List, Generic, Optional, TypeVar

from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr
from llama_index.core.instrumentation.span.base import BaseSpan

T = TypeVar("T", bound=BaseSpan)


class BaseSpanHandler(BaseModel, Generic[T]):
    open_spans: Dict[str, T] = Field(
        default_factory=dict, description="Dictionary of open spans."
    )
    completed_spans: List[T] = Field(
        default_factory=list, description="List of completed spans."
    )
    dropped_spans: List[T] = Field(
        default_factory=list, description="List of completed spans."
    )
    current_span_ids: Dict[Any, Optional[str]] = Field(
        default={}, description="Id of current spans in a given thread."
    )
    _lock: Optional[threading.Lock] = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        open_spans: Dict[str, T] = {},
        completed_spans: List[T] = [],
        dropped_spans: List[T] = [],
        current_span_ids: Dict[Any, str] = {},
    ):
        self._lock = None
        super().__init__(
            open_spans=open_spans,
            completed_spans=completed_spans,
            dropped_spans=dropped_spans,
            current_span_ids=current_span_ids,
        )

    def class_name(cls) -> str:
        """Class name."""
        return "BaseSpanHandler"

    @property
    def lock(self) -> threading.Lock:
        if self._lock is None:
            self._lock = threading.Lock()
        return self._lock

    @property
    def current_span_id(self) -> Optional[str]:
        current_thread = threading.get_ident()
        if current_thread in self.current_span_ids:
            return self.current_span_ids[current_thread]
        return None

    def set_current_span_id(self, value: str) -> None:
        current_thread = threading.get_ident()
        self.current_span_ids[current_thread] = value

    def span_enter(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Logic for entering a span."""
        if id_ in self.open_spans:
            pass  # should probably raise an error here
        else:
            # TODO: thread safe?
            span = self.new_span(
                id_=id_,
                bound_args=bound_args,
                instance=instance,
                parent_span_id=parent_id or self.current_span_id,
            )
            if span:
                with self.lock:
                    self.open_spans[id_] = span
                    self.set_current_span_id(id_)

    def span_exit(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Logic for exiting a span."""
        span = self.prepare_to_exit_span(
            id_=id_, bound_args=bound_args, instance=instance, result=result
        )
        if span:
            with self.lock:
                if self.current_span_id == id_:
                    self.set_current_span_id(self.open_spans[id_].parent_id)
                del self.open_spans[id_]
        if not self.open_spans:  # empty so flush
            with self.lock:
                self.set_current_span_id(None)

    def span_drop(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> None:
        """Logic for dropping a span i.e. early exit."""
        span = self.prepare_to_drop_span(
            id_=id_, bound_args=bound_args, instance=instance, err=err
        )
        if span:
            with self.lock:
                if self.current_span_id == id_:
                    self.set_current_span_id(self.open_spans[id_].parent_id)
                del self.open_spans[id_]

    @abstractmethod
    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        """Create a span.

        Subclasses of BaseSpanHandler should create the respective span type T
        and return it. Only NullSpanHandler should return a None here.
        """
        ...

    @abstractmethod
    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        """Logic for preparing to exit a span.

        Subclasses of BaseSpanHandler should return back the specific span T
        that is to be exited. If None is returned, then the span won't actually
        be exited.
        """
        ...

    @abstractmethod
    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        """Logic for preparing to drop a span.

        Subclasses of BaseSpanHandler should return back the specific span T
        that is to be dropped. If None is returned, then the span won't actually
        be dropped.
        """
        ...
