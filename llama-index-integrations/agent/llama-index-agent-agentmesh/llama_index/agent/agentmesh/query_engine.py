"""Trust-gated query engine for LlamaIndex."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import QueryBundle

from llama_index.agent.agentmesh.identity import CMVKIdentity
from llama_index.agent.agentmesh.trust import (
    TrustHandshake,
    TrustPolicy,
    TrustedAgentCard,
    TrustVerificationResult,
)


@dataclass
class DataAccessPolicy:
    """Policy for controlling data access in RAG pipelines."""

    allowed_collections: Optional[List[str]] = None
    denied_collections: Optional[List[str]] = None
    require_audit: bool = False
    max_results_per_query: int = 100
    allowed_metadata_filters: Optional[List[str]] = None
    require_identity: bool = True


@dataclass
class QueryAuditRecord:
    """Audit record for a query."""

    query_id: str
    timestamp: datetime
    invoker_did: Optional[str]
    query_text: str
    trust_verified: bool
    trust_score: float
    result_count: int
    warnings: List[str] = field(default_factory=list)


class TrustGatedQueryEngine(BaseQueryEngine):
    """
    Query engine with trust-based access control.

    This wrapper adds trust verification and access control
    to any LlamaIndex query engine.
    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        identity: CMVKIdentity,
        policy: Optional[TrustPolicy] = None,
        data_policy: Optional[DataAccessPolicy] = None,
    ):
        """
        Initialize trust-gated query engine.

        Args:
            query_engine: The underlying query engine
            identity: This engine's identity
            policy: Trust policy for verification
            data_policy: Data access policy

        """
        super().__init__(callback_manager=query_engine.callback_manager)
        self._query_engine = query_engine
        self._identity = identity
        self._policy = policy or TrustPolicy()
        self._data_policy = data_policy or DataAccessPolicy()
        self._handshake = TrustHandshake(identity, policy)
        self._audit_log: List[QueryAuditRecord] = []
        self._query_counter = 0

    def _generate_query_id(self) -> str:
        """Generate unique query ID."""
        self._query_counter += 1
        return f"q_{self._identity.did[-8:]}_{self._query_counter}"

    def _create_audit_record(
        self,
        query_text: str,
        invoker_card: Optional[TrustedAgentCard],
        verification: Optional[TrustVerificationResult],
        result_count: int = 0,
    ) -> QueryAuditRecord:
        """Create an audit record for a query."""
        return QueryAuditRecord(
            query_id=self._generate_query_id(),
            timestamp=datetime.now(timezone.utc),
            invoker_did=invoker_card.identity.did
            if invoker_card and invoker_card.identity
            else None,
            query_text=query_text[:500],  # Truncate for storage
            trust_verified=verification.trusted if verification else False,
            trust_score=verification.trust_score if verification else 0.0,
            result_count=result_count,
            warnings=verification.warnings if verification else [],
        )

    def verify_invoker(
        self,
        invoker_card: TrustedAgentCard,
        required_capabilities: Optional[List[str]] = None,
    ) -> TrustVerificationResult:
        """
        Verify an invoker before processing query.

        Args:
            invoker_card: The invoker's agent card
            required_capabilities: Required capabilities

        Returns:
            Verification result

        """
        return self._handshake.verify_peer(invoker_card, required_capabilities)

    def _query(
        self,
        query_bundle: QueryBundle,
        invoker_card: Optional[TrustedAgentCard] = None,
        **kwargs: Any,
    ) -> RESPONSE_TYPE:
        """
        Execute query with trust verification.

        Args:
            query_bundle: The query to execute
            invoker_card: Optional invoker card for verification
            **kwargs: Additional arguments

        Returns:
            Query response

        Raises:
            PermissionError: If trust verification fails

        """
        verification = None
        query_text = query_bundle.query_str

        # Verify invoker if required
        if self._policy.require_verification:
            if not invoker_card:
                if self._data_policy.require_identity:
                    raise PermissionError(
                        "Query requires invoker identity but none provided"
                    )
            else:
                verification = self.verify_invoker(invoker_card)
                if not verification.trusted and self._policy.block_unverified:
                    # Log blocked query
                    if self._policy.audit_queries:
                        record = self._create_audit_record(
                            query_text, invoker_card, verification
                        )
                        record.warnings.append("Query blocked due to trust failure")
                        self._audit_log.append(record)

                    raise PermissionError(f"Query rejected: {verification.reason}")

        # Execute the underlying query
        response = self._query_engine.query(query_bundle)

        # Audit if required
        if self._policy.audit_queries:
            # Count results (simplified)
            result_count = 1 if response else 0
            record = self._create_audit_record(
                query_text, invoker_card, verification, result_count
            )
            self._audit_log.append(record)

        return response

    async def _aquery(
        self,
        query_bundle: QueryBundle,
        invoker_card: Optional[TrustedAgentCard] = None,
        **kwargs: Any,
    ) -> RESPONSE_TYPE:
        """
        Async query with trust verification.

        Args:
            query_bundle: The query to execute
            invoker_card: Optional invoker card
            **kwargs: Additional arguments

        Returns:
            Query response

        """
        verification = None
        query_text = query_bundle.query_str

        if self._policy.require_verification:
            if not invoker_card:
                if self._data_policy.require_identity:
                    raise PermissionError(
                        "Query requires invoker identity but none provided"
                    )
            else:
                verification = self.verify_invoker(invoker_card)
                if not verification.trusted and self._policy.block_unverified:
                    raise PermissionError(f"Query rejected: {verification.reason}")

        response = await self._query_engine.aquery(query_bundle)

        if self._policy.audit_queries:
            result_count = 1 if response else 0
            record = self._create_audit_record(
                query_text, invoker_card, verification, result_count
            )
            self._audit_log.append(record)

        return response

    def query(
        self,
        str_or_query_bundle: str | QueryBundle,
        invoker_card: Optional[TrustedAgentCard] = None,
        **kwargs: Any,
    ) -> RESPONSE_TYPE:
        """
        Query with trust verification.

        Args:
            str_or_query_bundle: Query string or bundle
            invoker_card: Optional invoker card for verification
            **kwargs: Additional arguments

        Returns:
            Query response

        """
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(query_str=str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle

        return self._query(query_bundle, invoker_card=invoker_card, **kwargs)

    async def aquery(
        self,
        str_or_query_bundle: str | QueryBundle,
        invoker_card: Optional[TrustedAgentCard] = None,
        **kwargs: Any,
    ) -> RESPONSE_TYPE:
        """
        Async query with trust verification.

        Args:
            str_or_query_bundle: Query string or bundle
            invoker_card: Optional invoker card
            **kwargs: Additional arguments

        Returns:
            Query response

        """
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(query_str=str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle

        return await self._aquery(query_bundle, invoker_card=invoker_card, **kwargs)

    def get_audit_log(self) -> List[QueryAuditRecord]:
        """
        Get the query audit log.

        Returns:
            List of audit records

        """
        return self._audit_log.copy()

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log.clear()

    def get_audit_summary(self) -> Dict[str, Any]:
        """
        Get summary of audit activity.

        Returns:
            Audit summary dictionary

        """
        total = len(self._audit_log)
        verified = sum(1 for r in self._audit_log if r.trust_verified)

        return {
            "total_queries": total,
            "verified_queries": verified,
            "unverified_queries": total - verified,
            "verification_rate": verified / total if total > 0 else 1.0,
            "total_warnings": sum(len(r.warnings) for r in self._audit_log),
        }
