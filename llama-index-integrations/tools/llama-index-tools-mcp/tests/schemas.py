from typing import Literal, List

from pydantic import BaseModel, ConfigDict


MethodType = Literal["POST", "GET", "UPDATE", "DELETE"]
XY = List[str]


class TestName(BaseModel):
    name: str


class TestMethod(BaseModel):
    method: MethodType


class TestList(BaseModel):
    lst: List[int]


class StrictSchema(BaseModel):
    """Test schema with additionalProperties: false"""
    model_config = ConfigDict(extra='forbid')

    required_field: str
    optional_field: int | None = None
