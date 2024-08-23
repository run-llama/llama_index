from typing import Dict, TYPE_CHECKING


if TYPE_CHECKING:
    from .workflow import Workflow


class ServiceNotFoundError(Exception):
    """An error raised when the service manager couldn't find a certain service name."""


class ServiceManager:
    """An helper class to decouple how services are managed from the Workflow class.

    A Service is nothing more than a workflow instance attached to another workflow.
    The service is made available to the steps of the main workflow.
    """

    _services: Dict[str, "Workflow"] = {}

    def get(self, name: str) -> "Workflow":
        try:
            return self._services[name]
        except KeyError as e:
            msg = f"Service {name} not found"
            raise ServiceNotFoundError(msg)

    def add(self, name: str, service: "Workflow") -> None:
        self._services[name] = service
