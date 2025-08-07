from typing import Literal

from pydantic import BaseModel


type MethodType = Literal["POST", "GET", "UPDATE", "DELETE"]
type XY = list[str]


class TestName(BaseModel):
    name: str


class TestMethod(BaseModel):
    method: MethodType


class TestList(BaseModel):
    lst: list[int]
