from inspect import signature
from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_origin,
    get_args,
)
import datetime
import typing

from llama_index.core.bridge.pydantic import BaseModel, FieldInfo, create_model


def create_schema_from_function(
    name: str,
    func: Union[Callable[..., Any], Callable[..., Awaitable[Any]]],
    additional_fields: Optional[
        List[Union[Tuple[str, Type, Any], Tuple[str, Type]]]
    ] = None,
    ignore_fields: Optional[List[str]] = None,
) -> Type[BaseModel]:
    """
    Create schema from function.
    - Automatically adds json_schema_extra for basic Python types such as:
        - datetime.date -> format: "date"
        - datetime.datetime -> format: "date-time"
        - datetime.time -> format: "time"
    """
    fields = {}
    ignore_fields = ignore_fields or []
    params = signature(func).parameters

    for param_name in params:
        if param_name in ignore_fields:
            continue

        param_type = params[param_name].annotation
        param_default = params[param_name].default
        description = None
        json_schema_extra: dict[str, Any] = {}

        if get_origin(param_type) is typing.Annotated:
            args = get_args(param_type)
            param_type = args[0]

            if isinstance(args[1], str):
                description = args[1]
            elif isinstance(args[1], FieldInfo):
                description = args[1].description
                if args[1].json_schema_extra and isinstance(
                    args[1].json_schema_extra, dict
                ):
                    json_schema_extra.update(args[1].json_schema_extra)

        # Add format based on param_type
        if param_type == datetime.date:
            json_schema_extra.setdefault("format", "date")
        elif param_type == datetime.datetime:
            json_schema_extra.setdefault("format", "date-time")
        elif param_type == datetime.time:
            json_schema_extra.setdefault("format", "time")

        if param_type is params[param_name].empty:
            param_type = Any

        if param_default is params[param_name].empty:
            # Required field
            fields[param_name] = (
                param_type,
                FieldInfo(description=description, json_schema_extra=json_schema_extra),
            )
        elif isinstance(param_default, FieldInfo):
            # Field with pydantic.Field as default value
            fields[param_name] = (param_type, param_default)
        else:
            fields[param_name] = (
                param_type,
                FieldInfo(
                    default=param_default,
                    description=description,
                    json_schema_extra=json_schema_extra,
                ),
            )

    additional_fields = additional_fields or []
    for field_info in additional_fields:
        if len(field_info) == 3:
            field_info = cast(Tuple[str, Type, Any], field_info)
            field_name, field_type, field_default = field_info
            fields[field_name] = (field_type, FieldInfo(default=field_default))
        elif len(field_info) == 2:
            field_info = cast(Tuple[str, Type], field_info)
            field_name, field_type = field_info
            fields[field_name] = (field_type, FieldInfo())
        else:
            raise ValueError(
                f"Invalid additional field info: {field_info}. "
                "Must be a tuple of length 2 or 3."
            )

    return create_model(name, **fields)  # type: ignore
