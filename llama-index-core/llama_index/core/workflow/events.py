from typing import Any, Dict, Type, Union

from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr


class Event(BaseModel):
    """Base class for event types."""

    class Config:
        arbitrary_types_allowed = True


class DictLikeEvent(Event):
    """Base class for event types that mimics the interface of a dict."""

    _data: Dict[str, Any] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **params: Any):
        """__init__.

        NOTE: fields and private_attrs are pulled from params by name.
        """
        # pull out params for fields
        fields = {k: v for k, v in params.items() if k in self.__fields__}
        super().__init__(**fields)

        # set private attrs
        private_attrs = {
            k: v for k, v in params.items() if k in self.__private_attributes__
        }
        for private_attr, value in private_attrs.items():
            super().__setattr__(private_attr, value)

        # set underlying data dictionary
        data = {
            k: v
            for k, v in params.items()
            if k not in self.__fields__ and k not in self.__private_attributes__
        }

        self._data = data

    def __getattr__(self, __name: str) -> Any:
        if __name in self.__private_attributes__ or __name in self.__fields__:
            return super().__getattr__(__name)
        else:
            try:
                return self._data[__name]
            except KeyError:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{__name}'"
                )

    def __setattr__(self, name, value) -> None:
        if name in self.__private_attributes__ or name in self.__fields__:
            super().__setattr__(name, value)
        else:
            self._data.__setitem__(name, value)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> Dict[str, Any].keys:
        return self._data.keys()

    def values(self) -> Dict[str, Any].values:
        return self._data.values()

    def items(self) -> Dict[str, Any].items:
        return self._data.items()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Any:
        return iter(self._data)

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._data


class StartEvent(DictLikeEvent):
    """StartEvent is implicitly sent when a workflow runs."""


class StopEvent(DictLikeEvent):
    """EndEvent signals the workflow to stop."""

    result: Any = Field(default=None)

    def __init__(self, result: Any = None) -> None:
        # forces the user to provide a result
        super().__init__(result=result)


class CustomEvent(DictLikeEvent):
    """StartEvent is implicitly sent when a workflow runs."""

    _param1: str = PrivateAttr()


EventType = Type[Union[Event, DictLikeEvent]]
