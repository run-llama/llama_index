from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseKeyValStore(ABC):
    @abstractmethod
    def add(self, key: str, val: dict) -> None:
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[dict]:
        pass

    def get_all(self) -> Dict[str, dict]:
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        pass
