from dataclasses import dataclass
from typing import Optional


@dataclass
class VellumDeployment:
    """Represents a Deployment in Vellum.

    A Deployment is essentially an instance of an LLM Provider, Model,
    Prompt Template, and set of parameters (e.g. temperature).

    Monitoring in Vellum occurs at the Deployment level. You may have
    two Deployments with the same configuration in the event that
    you want to monitor traffic for each separately (i.e. if they serve
    different purposes / use-cases).
    """

    deployment_id: Optional[str]
    deployment_name: Optional[str]
    sandbox_id: Optional[str] = None

    def __post_init__(self) -> None:
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


@dataclass
class VellumCompiledPrompt:
    """Represents a compiled prompt from Vellum with all string substitutions,
    templating, etc. applied.
    """

    text: str
