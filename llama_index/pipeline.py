# from functools import
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Dict, List, Any


class PipelineSchema(BaseModel):
    """Class representing a component in a pipeline."""

    # TODO (jon-chuang): should name be an Enum to be more restrictive?
    name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    children: List["PipelineSchema"] = Field(default_factory=list)

    # TODO (jon-chuang): Handle DAG as opposed to tree.
    inputs: List["PipelineSchema"] = Field(default_factory=list)


class Pipeline(ABC):
    """
    A class that allows one to version LlamaIndex pipelines.

    For now, it can export model artifacts and prompt versions.

    In the future, it can be adapted towards declarative config.
    """

    @abstractmethod
    def get_schema(
        self,
        include_children: bool = True,
        omit_metadata: bool = False,
    ) -> PipelineSchema:
        pass

    def children(self) -> List[PipelineSchema]:
        raise NotImplementedError

    def inputs(self) -> List[PipelineSchema]:
        raise NotImplementedError


# TODO (jon-chuang): add automatic derivation via decorator.
# Not sure if necessary - so far a lot can be derived from base class.
