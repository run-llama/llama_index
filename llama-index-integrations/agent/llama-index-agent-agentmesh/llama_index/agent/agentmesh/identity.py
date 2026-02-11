"""
Cryptographic identity management for AgentMesh.

Uses Ed25519 for cryptographic operations.
"""

from __future__ import annotations

import base64
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature


@dataclass
class CMVKSignature:
    """A cryptographic signature from a CMVK identity."""

    algorithm: str = "CMVK-Ed25519"
    public_key: str = ""
    signature: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "public_key": self.public_key,
            "signature": self.signature,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CMVKSignature":
        timestamp_str = data.get("timestamp")
        return cls(
            algorithm=data.get("algorithm", "CMVK-Ed25519"),
            public_key=data.get("public_key", ""),
            signature=data.get("signature", ""),
            timestamp=(
                datetime.fromisoformat(timestamp_str)
                if timestamp_str
                else datetime.now(timezone.utc)
            ),
        )


@dataclass
class CMVKIdentity:
    """Cryptographic identity for an agent using CMVK scheme."""

    did: str
    agent_name: str
    public_key: str
    private_key: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def generate(
        cls, agent_name: str, capabilities: Optional[List[str]] = None
    ) -> "CMVKIdentity":
        """Generate a new CMVK identity with Ed25519 key pair."""
        seed = f"{agent_name}:{time.time_ns()}"
        did_hash = hashlib.sha256(seed.encode()).hexdigest()[:32]
        did = f"did:cmvk:{did_hash}"

        private_key_obj = ed25519.Ed25519PrivateKey.generate()
        public_key_obj = private_key_obj.public_key()
        private_key_b64 = base64.b64encode(private_key_obj.private_bytes_raw()).decode(
            "ascii"
        )
        public_key_b64 = base64.b64encode(public_key_obj.public_bytes_raw()).decode(
            "ascii"
        )

        return cls(
            did=did,
            agent_name=agent_name,
            public_key=public_key_b64,
            private_key=private_key_b64,
            capabilities=capabilities or [],
        )

    def sign(self, data: str) -> CMVKSignature:
        """Sign data with this identity's private key."""
        if not self.private_key:
            raise ValueError("Cannot sign without private key")

        private_key_bytes = base64.b64decode(self.private_key)
        private_key_obj = ed25519.Ed25519PrivateKey.from_private_bytes(
            private_key_bytes
        )
        signature_bytes = private_key_obj.sign(data.encode("utf-8"))
        signature_b64 = base64.b64encode(signature_bytes).decode("ascii")

        return CMVKSignature(public_key=self.public_key, signature=signature_b64)

    def verify_signature(self, data: str, signature: CMVKSignature) -> bool:
        """Verify a signature against this identity's public key."""
        if signature.public_key != self.public_key:
            return False

        try:
            public_key_bytes = base64.b64decode(self.public_key)
            public_key_obj = ed25519.Ed25519PublicKey.from_public_bytes(
                public_key_bytes
            )
            signature_bytes = base64.b64decode(signature.signature)
            public_key_obj.verify(signature_bytes, data.encode("utf-8"))
            return True
        except (InvalidSignature, ValueError):
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "did": self.did,
            "agent_name": self.agent_name,
            "public_key": self.public_key,
            "capabilities": self.capabilities,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CMVKIdentity":
        created_str = data.get("created_at")
        return cls(
            did=data["did"],
            agent_name=data["agent_name"],
            public_key=data["public_key"],
            capabilities=data.get("capabilities", []),
            created_at=(
                datetime.fromisoformat(created_str)
                if created_str
                else datetime.now(timezone.utc)
            ),
        )

    def public_identity(self) -> "CMVKIdentity":
        """Return a copy without the private key."""
        return CMVKIdentity(
            did=self.did,
            agent_name=self.agent_name,
            public_key=self.public_key,
            private_key=None,
            capabilities=self.capabilities.copy(),
            created_at=self.created_at,
        )
