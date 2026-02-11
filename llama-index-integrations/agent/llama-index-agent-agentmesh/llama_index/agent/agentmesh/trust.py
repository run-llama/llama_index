"""Trust verification protocols for AgentMesh."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from llama_index.agent.agentmesh.identity import CMVKIdentity, CMVKSignature


@dataclass
class TrustPolicy:
    """Policy configuration for trust verification."""

    require_verification: bool = True
    min_trust_score: float = 0.7
    allowed_capabilities: Optional[List[str]] = None
    audit_queries: bool = False
    block_unverified: bool = True
    cache_ttl_seconds: int = 900


@dataclass
class TrustVerificationResult:
    """Result of a trust verification operation."""

    trusted: bool
    trust_score: float
    reason: str
    verified_capabilities: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class TrustedAgentCard:
    """Agent card for discovery and verification."""

    name: str
    description: str
    capabilities: List[str]
    identity: Optional[CMVKIdentity] = None
    trust_score: float = 1.0
    card_signature: Optional[CMVKSignature] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _get_signable_content(self) -> str:
        content = {
            "name": self.name,
            "description": self.description,
            "capabilities": sorted(self.capabilities),
            "trust_score": self.trust_score,
            "identity_did": self.identity.did if self.identity else None,
            "identity_public_key": self.identity.public_key if self.identity else None,
        }
        return json.dumps(content, sort_keys=True, separators=(",", ":"))

    def sign(self, identity: CMVKIdentity) -> None:
        """Sign this card with the given identity."""
        self.identity = identity.public_identity()
        signable = self._get_signable_content()
        self.card_signature = identity.sign(signable)

    def verify_signature(self) -> bool:
        """Verify the card's signature."""
        if not self.identity or not self.card_signature:
            return False
        signable = self._get_signable_content()
        return self.identity.verify_signature(signable, self.card_signature)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "trust_score": self.trust_score,
            "metadata": self.metadata,
        }
        if self.identity:
            result["identity"] = self.identity.to_dict()
        if self.card_signature:
            result["card_signature"] = self.card_signature.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrustedAgentCard":
        identity = None
        if "identity" in data:
            identity = CMVKIdentity.from_dict(data["identity"])
        card_signature = None
        if "card_signature" in data:
            card_signature = CMVKSignature.from_dict(data["card_signature"])
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            capabilities=data.get("capabilities", []),
            identity=identity,
            trust_score=data.get("trust_score", 1.0),
            card_signature=card_signature,
            metadata=data.get("metadata", {}),
        )


class TrustHandshake:
    """Handles trust verification between agents."""

    def __init__(
        self,
        my_identity: CMVKIdentity,
        policy: Optional[TrustPolicy] = None,
    ):
        self.my_identity = my_identity
        self.policy = policy or TrustPolicy()
        self._verified_peers: Dict[str, tuple[TrustVerificationResult, datetime]] = {}
        self._cache_ttl = timedelta(seconds=self.policy.cache_ttl_seconds)

    def _get_cached_result(self, did: str) -> Optional[TrustVerificationResult]:
        if did in self._verified_peers:
            result, timestamp = self._verified_peers[did]
            if datetime.now(timezone.utc) - timestamp < self._cache_ttl:
                return result
            del self._verified_peers[did]
        return None

    def _cache_result(self, did: str, result: TrustVerificationResult) -> None:
        self._verified_peers[did] = (result, datetime.now(timezone.utc))

    def verify_peer(
        self,
        peer_card: TrustedAgentCard,
        required_capabilities: Optional[List[str]] = None,
        min_trust_score: Optional[float] = None,
    ) -> TrustVerificationResult:
        """Verify a peer agent's trustworthiness."""
        warnings: List[str] = []
        min_score = min_trust_score or self.policy.min_trust_score

        if peer_card.identity:
            cached = self._get_cached_result(peer_card.identity.did)
            if cached:
                return cached

        if not peer_card.identity:
            return TrustVerificationResult(
                trusted=False,
                trust_score=0.0,
                reason="No cryptographic identity provided",
            )

        if not peer_card.identity.did.startswith("did:cmvk:"):
            return TrustVerificationResult(
                trusted=False,
                trust_score=0.0,
                reason="Invalid DID format",
            )

        if not peer_card.verify_signature():
            return TrustVerificationResult(
                trusted=False,
                trust_score=0.0,
                reason="Card signature verification failed",
            )

        if peer_card.trust_score < min_score:
            return TrustVerificationResult(
                trusted=False,
                trust_score=peer_card.trust_score,
                reason=f"Trust score {peer_card.trust_score} below minimum {min_score}",
            )

        verified_caps = peer_card.capabilities.copy()
        if required_capabilities:
            missing = set(required_capabilities) - set(peer_card.capabilities)
            if missing:
                return TrustVerificationResult(
                    trusted=False,
                    trust_score=peer_card.trust_score,
                    reason=f"Missing required capabilities: {missing}",
                    verified_capabilities=verified_caps,
                )

        result = TrustVerificationResult(
            trusted=True,
            trust_score=peer_card.trust_score,
            reason="Verification successful",
            verified_capabilities=verified_caps,
            warnings=warnings,
        )

        self._cache_result(peer_card.identity.did, result)
        return result

    def clear_cache(self) -> None:
        self._verified_peers.clear()


@dataclass
class Delegation:
    """A delegation of capabilities."""

    delegator: str
    delegatee: str
    capabilities: List[str]
    signature: Optional[CMVKSignature] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None


class DelegationChain:
    """Manages a chain of trust delegations."""

    def __init__(self, root_identity: CMVKIdentity):
        self.root_identity = root_identity
        self.delegations: List[Delegation] = []
        self._known_identities: Dict[str, CMVKIdentity] = {
            root_identity.did: root_identity
        }

    def add_delegation(
        self,
        delegatee: TrustedAgentCard,
        capabilities: List[str],
        expires_in_hours: Optional[int] = None,
        delegator_identity: Optional[CMVKIdentity] = None,
    ) -> Delegation:
        """Add a delegation to the chain."""
        if not delegatee.identity:
            raise ValueError("Delegatee must have identity")

        delegator = delegator_identity or self.root_identity
        delegatee_did = delegatee.identity.did

        expires_at = None
        if expires_in_hours:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)

        delegation_data = json.dumps(
            {
                "delegator": delegator.did,
                "delegatee": delegatee_did,
                "capabilities": sorted(capabilities),
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
            sort_keys=True,
        )

        signature = delegator.sign(delegation_data)

        delegation = Delegation(
            delegator=delegator.did,
            delegatee=delegatee_did,
            capabilities=capabilities,
            signature=signature,
            expires_at=expires_at,
        )

        self.delegations.append(delegation)
        self._known_identities[delegatee_did] = delegatee.identity
        return delegation

    def verify(self) -> bool:
        """Verify the entire delegation chain."""
        if not self.delegations:
            return True

        for i, delegation in enumerate(self.delegations):
            if delegation.expires_at and delegation.expires_at < datetime.now(
                timezone.utc
            ):
                return False

            if not delegation.signature:
                return False

            delegator_identity = self._known_identities.get(delegation.delegator)
            if not delegator_identity:
                return False

            delegation_data = json.dumps(
                {
                    "delegator": delegation.delegator,
                    "delegatee": delegation.delegatee,
                    "capabilities": sorted(delegation.capabilities),
                    "expires_at": delegation.expires_at.isoformat()
                    if delegation.expires_at
                    else None,
                },
                sort_keys=True,
            )

            if not delegator_identity.verify_signature(
                delegation_data, delegation.signature
            ):
                return False

            if i > 0:
                prev_delegation = self.delegations[i - 1]
                if delegation.delegator != prev_delegation.delegatee:
                    return False

        return True
