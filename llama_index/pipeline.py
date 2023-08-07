# from functools import
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any


@dataclass
class PipelineSchema:
    """Class representing a component in a pipeline."""

    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["PipelineSchema"] = field(default_factory=list)

    def json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class Pipeline(ABC):
    """
    A class that allows one to version LlamaIndex pipelines.

    For now, it can export model artifacts and prompt versions.

    In the future, it can be adapted towards declarative config.
    """

    @abstractmethod
    def schema(
        self,
        include_children: bool = True,
        omit_metadata: bool = False,
    ) -> PipelineSchema:
        pass

    def children(self) -> List[PipelineSchema]:  # Which?
        raise NotImplementedError


# TODO (jon-chuang): add automatic derivation via decorator.
# Not sure if necessary - so far a lot can be derived from base class.
