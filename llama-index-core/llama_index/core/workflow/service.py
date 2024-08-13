from typing import Dict, TYPE_CHECKING


if TYPE_CHECKING:
    from .workflow import Workflow


class ServiceManager:
    _services: Dict[str, "Workflow"] = {}

    def get(self, name: str) -> "Workflow":
        return self._services[name]

    def add(self, name: str, service: "Workflow"):
        self._services[name] = service
