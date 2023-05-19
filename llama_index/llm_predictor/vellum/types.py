from dataclasses import dataclass
from typing import Optional


@dataclass
class VellumDeployment:
    deployment_id: Optional[str]
    deployment_name: Optional[str]
    sandbox_id: Optional[str] = None

    def __post_init__(self):
        if not self.deployment_id and not self.deployment_name:
            raise ValueError(
                "Either `deployment_id` or `deployment_name` must be provided."
            )

    @property
    def deployment_url(self) -> Optional[str]:
        if not self.deployment_id:
            return None

        return f"https://vellum.ai/deployments/{self.deployment_id}"

    @property
    def sandbox_url(self) -> Optional[str]:
        if not self.sandbox_id:
            return None

        return f"https://vellum.ai/playground/sandbox/{self.sandbox_id}"
