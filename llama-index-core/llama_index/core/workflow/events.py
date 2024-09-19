from typing import Any, Dict, Type
from _collections_abc import dict_keys, dict_items, dict_values

from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr, ConfigDict


class Event(BaseModel):
    """Base class for event types that mimics dict interface.

    PrivateAttr:
        _data (Dict[str, Any]): Underlying Python dict.

    Examples:
        Basic example usage

        ```python
        from llama_index.core.workflows.events import Event

        evt = Event(a=1, b=2)

        # can use dot access to get values of `a` and `b`
        print((evt.a, evt.b))

        # can also set the attrs
        evt.a = 2
        ```

        Custom event with additional Fields/PrivateAttr

        ```python
        from llama_index.core.workflows.events import Event
        from llama_index.core.bridge.pydantic import Field, PrivateAttr

        class CustomEvent(Event):
            field_1: int = Field(description="my custom field")
            _private_attr_1: int = PrivateAttr()

        evt = CustomEvent(a=1, b=2, field_1=3, _private_attr_1=4)

        # `field_1` and `_private_attr_1` get set as they do with Pydantic BaseModel
        print(evt.field_1)
        print(evt._private_attr_1)

        # `a` and `b` get set in the underliying dict, namely `evt._data`
        print((evt.a, evt.b))
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _data: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, **params: Any):
        """__init__.

        NOTE: fields and private_attrs are pulled from params by name.
        """
        # extract and set fields, private attrs and remaining shove in _data
        fields = {}
        private_attrs = {}
        data = {}
        for k, v in params.items():
            if k in self.model_fields:
                fields[k] = v
            elif k in self.__private_attributes__:
                private_attrs[k] = v
            else:
                data[k] = v
        super().__init__(**fields)
        for private_attr, value in private_attrs.items():
            super().__setattr__(private_attr, value)
        self._data = data

    def __getattr__(self, __name: str) -> Any:
        if __name in self.__private_attributes__ or __name in self.model_fields:
            return super().__getattr__(__name)  # type: ignore
        else:
            try:
                return self._data[__name]
            except KeyError:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{__name}'"
                )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__private_attributes__ or name in self.model_fields:
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

    def keys(self) -> "dict_keys[str, Any]":
        return self._data.keys()

    def values(self) -> "dict_values[str, Any]":
        return self._data.values()

    def items(self) -> "dict_items[str, Any]":
        return self._data.items()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Any:
        return iter(self._data)

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._data


class StartEvent(Event):
    """StartEvent is implicitly sent when a workflow runs."""


class StopEvent(Event):
    """EndEvent signals the workflow to stop."""

    result: Any = Field(default=None)

    def __init__(self, result: Any = None) -> None:
        # forces the user to provide a result
        super().__init__(result=result)


EventType = Type[Event]
