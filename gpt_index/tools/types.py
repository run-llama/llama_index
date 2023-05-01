from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ToolMetadata:
    description: str
    name: Optional[str] = None


class BaseTool:
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        pass

    @abstractmethod
    def __call__(self, input: Any) -> None:
        pass
