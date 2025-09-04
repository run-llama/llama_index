from typing import Literal, List

from pydantic import BaseModel


MethodType = Literal["POST", "GET", "UPDATE", "DELETE"]
XY = List[str]


class TestName(BaseModel):
    name: str


class TestMethod(BaseModel):
    method: MethodType


class TestList(BaseModel):
    lst: List[int]
