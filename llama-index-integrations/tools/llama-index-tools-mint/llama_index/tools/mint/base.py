"""MINT Protocol tool spec for LlamaIndex."""

from typing import Any, List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class MintToolSpec(BaseToolSpec):
    """MINT Protocol tool spec.

    Gives an agent universal *work attestation*: attest a completed unit of work
    to a tamper-evident, on-chain (Solana) record, verify any actor's trust
    profile, discover trusted agents/services by capability, and rate or
    recommend other actors to build portable reputation across the agent economy.

    Thin wrapper over the ``mint-attest`` SDK
    (https://pypi.org/project/mint-attest/). All blockchain interaction happens
    server-side, so the agent never touches a wallet or signs a transaction — every
    method is a plain authenticated HTTPS call. The same service is also available
    as an MCP server (https://smithery.ai/server/@foundrynet/mint-protocol).
    """

    spec_functions = [
        "attest_work",
        "verify_trust",
        "discover_actors",
        "rate_attestation",
        "recommend_actor",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the MINT client.

        Args:
            api_key (Optional[str]): Your MINT ``fnet_`` API key. Falls back to the
                ``MINT_API_KEY`` environment variable. Get one at
                https://mint.foundrynet.io.
            endpoint (Optional[str]): Override the MINT server endpoint (defaults to
                the hosted MINT service / ``MINT_ENDPOINT``).
            name (Optional[str]): Display name for this agent's MINT identity, used
                when the agent auto-registers on its first attestation.
            capabilities (Optional[List[str]]): Capability tags advertised for this
                agent in the discovery directory, e.g. ``["code_review", "rag"]``.

        """
        from mint_attest import MintClient

        self.client = MintClient(
            api_key=api_key,
            endpoint=endpoint,
            name=name,
            capabilities=capabilities,
        )

    def attest_work(
        self,
        work_type: str,
        summary: Optional[str] = None,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        duration_seconds: Optional[float] = None,
    ) -> dict:
        """
        Attest a completed unit of work.

        Anchors a tamper-evident record on Solana mainnet and returns a receipt
        containing the attestation id, data hash, transaction signature, and a
        public ``verify_url``. The agent auto-registers on its first attestation.

        Args:
            work_type (str): What kind of work was done, e.g. ``"code_review"``,
                ``"generation"``, ``"data_extraction"``.
            summary (Optional[str]): Short human-readable summary of the work.
            input_data (Optional[Any]): The work's input — hashed client-side, never
                sent in the clear.
            output_data (Optional[Any]): The work's output — hashed client-side.
            duration_seconds (Optional[float]): How long the work took.

        Returns:
            dict: The attestation receipt (attestation_id, data_hash,
            tx_signature, verify_url, ...).

        """
        receipt = self.client.attest(
            work_type=work_type,
            summary=summary,
            input_data=input_data,
            output_data=output_data,
            duration_seconds=duration_seconds,
        )
        return receipt.raw or {"attestation_id": receipt.attestation_id}

    def verify_trust(
        self,
        mint_id: Optional[str] = None,
        actor_name: Optional[str] = None,
    ) -> dict:
        """
        Look up an actor's trust profile.

        Args:
            mint_id (Optional[str]): The actor's MINT id. Defaults to this agent's
                own identity when neither argument is given.
            actor_name (Optional[str]): Look the actor up by name instead of id.

        Returns:
            dict: The trust profile (trust_score, total_attestations, avg_rating,
            recommendations, work_types, ...).

        """
        return self.client.verify(mint_id=mint_id, actor_name=actor_name).raw

    def discover_actors(
        self,
        capability: Optional[str] = None,
        actor_type: Optional[str] = None,
        min_trust: float = 0,
        limit: int = 10,
    ) -> List[dict]:
        """
        Search the directory for trusted actors, best first.

        Args:
            capability (Optional[str]): Capability or keyword to search for, e.g.
                ``"telemetry normalization"``.
            actor_type (Optional[str]): Filter by type — ``"ai_agent"``,
                ``"machine"``, ``"iot_device"``, or ``"service"``.
            min_trust (float): Only return actors at or above this trust score
                (0-100).
            limit (int): Maximum number of results (1-50).

        Returns:
            List[dict]: Matching actors, ranked by trust score.

        """
        results = self.client.discover(
            capability=capability,
            actor_type=actor_type,
            min_trust=min_trust,
            limit=limit,
        )
        return [d.raw for d in results]

    def rate_attestation(
        self,
        attestation_id: str,
        rated_mint_id: str,
        score: int,
        comment: Optional[str] = None,
    ) -> dict:
        """
        Rate a completed attestation 1-5, updating the rated actor's trust score.

        Args:
            attestation_id (str): The attestation being rated.
            rated_mint_id (str): The actor that did the work.
            score (int): A rating from 1 to 5.
            comment (Optional[str]): Optional free-text feedback.

        Returns:
            dict: The rating result.

        """
        return self.client.rate(
            attestation_id=attestation_id,
            rated_mint_id=rated_mint_id,
            score=score,
            comment=comment,
        ).raw

    def recommend_actor(
        self,
        recommended_mint_id: str,
        context: str,
        score: int,
        note: Optional[str] = None,
    ) -> dict:
        """
        Endorse another actor in a named context 1-5, updating their trust score.

        Args:
            recommended_mint_id (str): The actor you're endorsing.
            context (str): What you're endorsing them for, e.g.
                ``"cross-oem normalization"``.
            score (int): A score from 1 to 5.
            note (Optional[str]): Optional free-text note.

        Returns:
            dict: The recommendation result.

        """
        return self.client.recommend(
            recommended_mint_id=recommended_mint_id,
            context=context,
            score=score,
            note=note,
        ).raw
