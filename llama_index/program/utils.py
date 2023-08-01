"""Program utils."""

from pydantic import BaseModel, Field
from typing import Type, List


def create_list_model(base_cls: Type[BaseModel]) -> Type[BaseModel]:
    """Create a list version of an existing Pydantic object."""

    # NOTE: this is directly taken from
    # https://github.com/jxnl/openai_function_call/blob/main/examples/streaming_multitask/streaming_multitask.py
    # all credits go to the openai_function_call repo

    name = f"{base_cls.__name__}List"
    list_items = (
        List[base_cls],
        Field(default_factory=list, repr=False, description=f"List of {base_cls.__name__} items"),
    )

    new_cls = 
