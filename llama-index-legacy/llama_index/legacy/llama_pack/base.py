"""Llama pack class."""

from abc import abstractmethod
from typing import Any, Dict


class BaseLlamaPack:
    @abstractmethod
    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run."""
