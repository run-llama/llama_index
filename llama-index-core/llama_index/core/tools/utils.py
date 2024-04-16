from inspect import signature
from typing import Any, Callable, List, Optional, Tuple, Type, Union, cast, Dict

from llama_index.core.bridge.pydantic import BaseModel, FieldInfo, create_model, Field

import docstring_parser


def create_schema_from_function(
    name: str,
    func: Callable[..., Any],
    additional_fields: Optional[
        List[Union[Tuple[str, Type, Any], Tuple[str, Type]]]
    ] = None,
) -> Type[BaseModel]:
    """Create schema from function."""
    fields: Dict[str, FieldInfo] = {}
    params = signature(func).parameters
    doc = docstring_parser.parse(func.__doc__)

    params_doc = {param.arg_name: param for param in doc.params}
    for param_name in params:
        param_type = params[param_name].annotation
        param_default = params[param_name].default
        param_desc = (
            params_doc[param_name].description if param_name in params_doc else None
        )

        if param_type is params[param_name].empty:
            param_type = Any

        if param_default is params[param_name].empty:
            # Required field
            field_info = Field()
        elif isinstance(param_default, FieldInfo):
            # Field with pydantic.FieldInfo as default value
            field_info = param_default
        else:
            field_info = Field(default=param_default)
        field_info = cast(FieldInfo, field_info)

        if param_desc:
            field_info.description = param_desc

        fields[param_name] = (param_type, field_info)

    additional_fields = additional_fields or []
    for field_info in additional_fields:
        if len(field_info) == 3:
            field_info = cast(Tuple[str, Type, Any], field_info)
            field_name, field_type, field_default = field_info
            fields[field_name] = (field_type, FieldInfo(default=field_default))
        elif len(field_info) == 2:
            # Required field has no default value
            field_info = cast(Tuple[str, Type], field_info)
            field_name, field_type = field_info
            fields[field_name] = (field_type, FieldInfo())
        else:
            raise ValueError(
                f"Invalid additional field info: {field_info}. "
                "Must be a tuple of length 2 or 3."
            )

    return create_model(name, **fields)  # type: ignore
