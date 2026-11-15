from inspect import Parameter, signature
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
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


def _merge_field_fallbacks(
    field: FieldInfo,
    description: Optional[str],
    json_schema_extra: Dict[str, Any],
) -> FieldInfo:
    """
    Apply description/json_schema_extra fallbacks to a field without
    dropping constraints already declared on it (e.g. `pattern`, numeric
    bounds). Only sets an attribute when the field doesn't already carry one,
    and merges rather than replaces `json_schema_extra`.
    """
    if field.description is None and description is not None:
        field = FieldInfo.merge_field_infos(field, description=description)
    if json_schema_extra:
        extra = field.json_schema_extra
        merged_extra = dict(extra) if isinstance(extra, dict) else {}
        merged_extra.update(json_schema_extra)
        if merged_extra != extra:
            field = FieldInfo.merge_field_infos(field, json_schema_extra=merged_extra)
    return field


def create_schema_from_function(
    name: str,
    func: Union[Callable[..., Any], Callable[..., Awaitable[Any]]],
    additional_fields: Optional[
        List[Union[Tuple[str, Type, Any], Tuple[str, Type]]]
    ] = None,
    ignore_fields: Optional[List[str]] = None,
    param_descriptions: Optional[Dict[str, str]] = None,
) -> Type[BaseModel]:
    """
    Create schema from function.
    - Automatically adds json_schema_extra for basic Python types such as:
        - datetime.date -> format: "date"
        - datetime.datetime -> format: "date-time"
        - datetime.time -> format: "time"
    - `param_descriptions` supplies fallback descriptions (e.g. parsed from the
      function's docstring) for parameters that don't carry one of their own.
      These must be passed in here rather than assigned to `model_fields` after
      the fact: `model_json_schema()` is derived from the core schema frozen at
      `create_model()` time, so a later mutation never reaches the schema.
    """
    fields = {}
    ignore_fields = ignore_fields or []
    param_descriptions = param_descriptions or {}
    params = signature(func).parameters

    for param_name in params:
        if param_name in ignore_fields:
            continue

        if params[param_name].kind in (
            Parameter.VAR_POSITIONAL,
            Parameter.VAR_KEYWORD,
        ):
            continue

        param_type = params[param_name].annotation
        param_default = params[param_name].default
        description = None
        json_schema_extra: dict[str, Any] = {}
        annotated_field: Optional[FieldInfo] = None

        if get_origin(param_type) is typing.Annotated:
            args = get_args(param_type)
            param_type = args[0]

            # Preserve any pydantic FieldInfo declared in the Annotated
            # metadata (e.g. `Annotated[str, Field(pattern=...)]`): dropping
            # it here would lose its JSON schema constraints (pattern,
            # min_length, numeric bounds, etc.) in the tool schema sent to
            # the provider.
            field_infos = [arg for arg in args[1:] if isinstance(arg, FieldInfo)]
            str_descriptions = [arg for arg in args[1:] if isinstance(arg, str)]
            if field_infos:
                annotated_field = FieldInfo.merge_field_infos(*field_infos)
                description = annotated_field.description
                if description is None and str_descriptions:
                    description = str_descriptions[0]
            elif str_descriptions:
                description = str_descriptions[0]

        if description is None:
            description = param_descriptions.get(param_name)

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
            if annotated_field is not None:
                annotated_field = _merge_field_fallbacks(
                    annotated_field, description, json_schema_extra
                )
                fields[param_name] = (param_type, annotated_field)
            else:
                fields[param_name] = (
                    param_type,
                    FieldInfo(
                        description=description, json_schema_extra=json_schema_extra
                    ),
                )
        elif isinstance(param_default, FieldInfo):
            # Field with pydantic.Field as default value
            if annotated_field is not None:
                param_default = FieldInfo.merge_field_infos(
                    param_default, annotated_field
                )
            if param_default.description is None and description is not None:
                # Merge rather than assign: pydantic builds the field from the
                # attributes the FieldInfo records as explicitly set, so a plain
                # `param_default.description = ...` would never reach the schema
                # (and would edit the caller's instance).
                param_default = FieldInfo.merge_field_infos(
                    param_default, description=description
                )
            fields[param_name] = (param_type, param_default)
        else:
            base_field = FieldInfo(
                default=param_default,
                description=description,
                json_schema_extra=json_schema_extra,
            )
            if annotated_field is not None:
                base_field = FieldInfo.merge_field_infos(base_field, annotated_field)
            fields[param_name] = (param_type, base_field)

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
