import inspect
from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.instrumentation.span.base import BaseSpan

T = TypeVar("T", bound=BaseSpan)


class BaseSpanHandler(BaseModel, Generic[T]):
    open_spans: Dict[str, T] = Field(
        default_factory=dict, description="Dictionary of open spans."
    )
    current_span_id: Optional[str] = Field(
        default=None, description="Id of current span."
    )

    class Config:
        arbitrary_types_allowed = True

    def class_name(cls) -> str:
        """Class name."""
        return "BaseSpanHandler"

    def span_enter(
        self,
        *args: Any,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
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
                parent_span_id=self.current_span_id,
            )
            if span:
                self.open_spans[id_] = span
                self.current_span_id = id_

    def span_exit(
        self,
        *args: Any,
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
            if self.current_span_id == id_:
                self.current_span_id = self.open_spans[id_].parent_id
            del self.open_spans[id_]

    def span_drop(
        self,
        *args: Any,
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
            if self.current_span_id == id_:
                self.current_span_id = self.open_spans[id_].parent_id
            del self.open_spans[id_]

    @abstractmethod
    def new_span(
        self,
        *args: Any,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        """Create a span."""
        ...

    @abstractmethod
    def prepare_to_exit_span(
        self,
        *args: Any,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        """Logic for preparing to exit a span."""
        ...

    @abstractmethod
    def prepare_to_drop_span(
        self,
        *args: Any,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        """Logic for preparing to drop a span."""
        ...
