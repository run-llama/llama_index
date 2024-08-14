from typing import Dict, TYPE_CHECKING


if TYPE_CHECKING:
    from .workflow import Workflow


class ServiceManager:
    _services: Dict[str, "Workflow"] = {}

    def get(self, name: str) -> "Workflow":
        return self._services[name]

    def add(self, name: str, service: "Workflow"):
        self._services[name] = service

    @classmethod
    def from_workflows(cls, **workflows: "Workflow") -> "ServiceManager":
        sm = cls()
        for name, wf in workflows.items():
            sm.add(name, wf)
        return sm
