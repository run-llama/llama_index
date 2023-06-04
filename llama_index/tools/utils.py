"""Tool utilies."""
from typing import Callable, Any, Optional, List, Tuple, Type
from pydantic import BaseModel, create_model
from inspect import signature


def create_schema_from_function(
    name: str,
    func: Callable[..., Any],
    additional_fields: Optional[List[Tuple[str, Type, Any]]] = None,
) -> Type[BaseModel]:
    """Create schema from function."""
    # NOTE: adapted from langchain.tools.base
    fields = {}
    params = signature(func).parameters
    for param_name in params.keys():
        param_type = params[param_name].annotation
        param_default = params[param_name].default
        if param_default is params[param_name].empty:
            param_default = None
        fields[param_name] = (param_type, param_default)

    additional_fields = additional_fields or []
    for field_name, field_type, field_default in additional_fields:
        fields[field_name] = (field_type, field_default)

    return create_model(name, **fields)  # type: ignore
