"""Agent Module tool spec."""

from typing import Optional

import requests

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class AgentModuleToolSpec(BaseToolSpec):
    """
    Agent Module tool spec for EU AI Act compliance knowledge.

    Provides deterministic EU AI Act compliance knowledge from Agent Module.
    Returns binary logic gates and specific statutory citations. No probabilistic
    inference — all records have confidence_required: 1.0.

    Args:
        am_key: Agent Module API key (X-Agent-Module-Key header).
                Get a free 24-hour trial key via POST https://api.agent-module.dev/api/trial
                or by calling the MCP tool ``get_trial_key``.
        vertical: Knowledge vertical to query. Default: "ethics".
        timeout: Request timeout in seconds. Default: 10.

    """

    spec_functions = [
        "query_module",
        "query_fria",
        "query_prohibited_practices",
        "query_high_risk_classification",
        "query_risk_management",
        "query_conformity_assessment",
        "query_gpai_obligations",
    ]

    def __init__(
        self,
        am_key: Optional[str] = None,
        vertical: str = "ethics",
        timeout: int = 10,
    ) -> None:
        self.am_key = am_key
        self.vertical = vertical
        self.timeout = timeout

    def _build_headers(self) -> dict:
        if self.am_key:
            return {"X-Agent-Module-Key": self.am_key}
        return {}

    def _to_node_id(self, module: str, vertical: str) -> str:
        """Convert module identifier to full node ID. ETH_013 -> node:ethics:eth013."""
        node_key = module.lower().replace("_", "").replace("-", "")
        return f"node:{vertical}:{node_key}"

    def _get(self, module: str, vertical: Optional[str] = None) -> str:
        """Execute synchronous GET request to Agent Module API."""
        vert = vertical or self.vertical
        try:
            response = requests.get(
                "https://api.agent-module.dev/api/demo",
                params={"vertical": vert, "node": self._to_node_id(module, vert)},
                headers=self._build_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.text
        except requests.HTTPError as e:
            return f"Agent Module HTTP error ({e.response.status_code}): {e}"
        except requests.RequestException as e:
            return f"Agent Module connection error: {e}"

    def query_module(self, module_id: str, vertical: Optional[str] = None) -> str:
        """
        Query any Agent Module knowledge node by module ID.

        Use this for direct module lookups when you know the exact identifier.

        Args:
            module_id: Module identifier in ETH_XXX format.
                       Examples: "ETH_016" (prohibited practices),
                       "ETH_015" (high-risk classification), "ETH_017" (risk management),
                       "ETH_013" (conformity assessment), "ETH_020" (GPAI obligations).
                       ETH_021 (FRIA) and above require a membership key.
            vertical: Knowledge vertical to query. Defaults to instance vertical ("ethics").

        Returns:
            JSON string with knowledge records, logic gates, and statutory citations.

        """
        return self._get(module_id, vertical)

    def query_fria(self) -> str:
        """
        Query FRIA (Fundamental Rights Impact Assessment) requirements.

        Retrieves Art. 27 obligations for deployers of high-risk AI systems.
        August 2026 enforcement deadline. Covers: scope triggers, assessment
        methodology, documentation requirements, and competent authority obligations.

        Requires a membership key — ETH_021 is not accessible in demo mode.
        Get a free 24-hour trial key via POST https://api.agent-module.dev/api/trial
        or by calling the MCP tool get_trial_key.

        Returns:
            JSON with ETH_021 FRIA logic gates and Art. 27 statutory citations.

        """
        return self._get("ETH_021")

    def query_prohibited_practices(self) -> str:
        """
        Query prohibited AI practices under Art. 5 of the EU AI Act.

        Returns deterministic binary gates for the highest-penalty tier (€35M or
        7% of global annual turnover). Covers: subliminal manipulation, exploitation
        of vulnerabilities, social scoring, real-time biometric surveillance,
        and emotion recognition in workplace/education.

        Returns:
            JSON with ETH_016 prohibited practice logic gates and Art. 5 citations.

        """
        return self._get("ETH_016")

    def query_high_risk_classification(self) -> str:
        """
        Query high-risk AI classification criteria under Annex III.

        Returns classification logic gates for 8 high-risk categories under Art. 6
        and Annex III: biometric identification, critical infrastructure, education,
        employment, essential services, law enforcement, migration, justice.

        Returns:
            JSON with ETH_015 classification logic gates and Annex III citations.

        """
        return self._get("ETH_015")

    def query_risk_management(self) -> str:
        """
        Query risk management system obligations under Art. 9.

        Returns iterative risk management requirements for providers of high-risk AI:
        risk identification, risk estimation, risk evaluation, risk mitigation,
        residual risk acceptance, and post-market monitoring integration.

        Returns:
            JSON with ETH_017 risk management logic gates and Art. 9 citations.

        """
        return self._get("ETH_017")

    def query_conformity_assessment(self) -> str:
        """
        Query conformity assessment procedures under Art. 43.

        Returns logic gates for CE marking, notified body requirements,
        full internal control vs. third-party assessment paths,
        and Annex VII documentation obligations.

        Returns:
            JSON with ETH_013 conformity assessment logic gates and Art. 43 citations.

        """
        return self._get("ETH_013")

    def query_gpai_obligations(self) -> str:
        """
        Query GPAI (General Purpose AI) model obligations under Art. 53-55.

        Returns obligations for providers of general-purpose AI models including
        technical documentation, copyright transparency, energy consumption
        reporting, and systemic risk assessment for high-capability models.

        Returns:
            JSON with ETH_020 GPAI logic gates and Art. 53-55 citations.

        """
        return self._get("ETH_020")
