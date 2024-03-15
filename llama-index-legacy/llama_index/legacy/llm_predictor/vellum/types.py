from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class VellumRegisteredPrompt:
    deployment_id: str
    deployment_name: str
    model_version_id: str
    sandbox_id: str | None = None
    sandbox_snapshot_id: str | None = None
    prompt_id: str | None = None

    @property
    def deployment_url(self) -> str | None:
        if not self.deployment_id:
            return None

        return f"https://app.vellum.ai/deployments/{self.deployment_id}"

    @property
    def sandbox_url(self) -> str | None:
        if not self.sandbox_id:
            return None

        url = f"https://app.vellum.ai/playground/sandbox/{self.sandbox_id}"
        if not self.sandbox_snapshot_id:
            return url

        url += f"?snapshotId={self.sandbox_snapshot_id}"

        return url


@dataclass
class VellumCompiledPrompt:
    """Represents a compiled prompt from Vellum with all string substitutions,
    templating, etc. applied.
    """

    text: str
    num_tokens: int
