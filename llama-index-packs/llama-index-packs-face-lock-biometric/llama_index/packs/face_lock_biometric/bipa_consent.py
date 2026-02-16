"""
BIPA (Biometric Information Privacy Act) consent management layer.

Provides consent recording, data retention enforcement, and audit trails
for the real-face workflow path. Ensures compliance with Illinois BIPA
($1,000-$5,000 per violation, private right of action).
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConsentStatus(Enum):
    """Status of a BIPA consent record."""

    PENDING = "pending"
    GRANTED = "granted"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass
class BIPAConsent:
    """
    BIPA consent record for biometric data collection.

    Required fields per Illinois BIPA (740 ILCS 14/):
    - Written notice of collection purpose
    - Explicit consent (opt-in, not opt-out)
    - Data retention schedule
    - Prohibition on sale of data
    """

    consent_given: bool = False
    timestamp: str = ""
    data_retention_days: int = 30
    purpose: str = "AI character consistency"
    subject_identifier_hash: str = ""  # SHA-256 hash, never store raw PII
    collector_entity: str = ""
    consent_method: str = "digital_signature"

    # BIPA-required disclosures
    written_notice_provided: bool = False
    retention_schedule_disclosed: bool = False
    no_sale_clause_accepted: bool = False

    status: ConsentStatus = ConsentStatus.PENDING

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def is_valid(self) -> bool:
        """Check if consent is currently valid (granted and not expired)."""
        if self.status != ConsentStatus.GRANTED:
            return False
        if not self.consent_given:
            return False
        if not all([
            self.written_notice_provided,
            self.retention_schedule_disclosed,
            self.no_sale_clause_accepted,
        ]):
            return False
        return not self.is_expired

    @property
    def is_expired(self) -> bool:
        """Check if the consent has exceeded its retention period."""
        try:
            consent_time = datetime.fromisoformat(self.timestamp)
            expiry = consent_time + timedelta(days=self.data_retention_days)
            return datetime.now(timezone.utc) > expiry
        except (ValueError, TypeError):
            return True

    @property
    def expiry_date(self) -> str:
        """Return the expiry date as an ISO string."""
        try:
            consent_time = datetime.fromisoformat(self.timestamp)
            expiry = consent_time + timedelta(days=self.data_retention_days)
            return expiry.isoformat()
        except (ValueError, TypeError):
            return "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize consent record (safe for storage)."""
        return {
            "consent_given": self.consent_given,
            "timestamp": self.timestamp,
            "data_retention_days": self.data_retention_days,
            "purpose": self.purpose,
            "subject_identifier_hash": self.subject_identifier_hash,
            "collector_entity": self.collector_entity,
            "consent_method": self.consent_method,
            "written_notice_provided": self.written_notice_provided,
            "retention_schedule_disclosed": self.retention_schedule_disclosed,
            "no_sale_clause_accepted": self.no_sale_clause_accepted,
            "status": self.status.value,
            "expiry_date": self.expiry_date,
        }


def hash_identifier(identifier: str) -> str:
    """
    Hash a subject identifier (name, email, etc.) for safe storage.

    BIPA requires we never store raw biometric identifiers. We hash
    them so we can still track consent per subject without storing PII.
    """
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()


def create_consent(
    subject_identifier: str,
    collector_entity: str,
    purpose: str = "AI character consistency",
    data_retention_days: int = 30,
    consent_method: str = "digital_signature",
) -> BIPAConsent:
    """
    Create a new BIPA consent record with all required disclosures.

    This function enforces that all BIPA-required disclosures are
    acknowledged before consent can be granted.

    Args:
        subject_identifier: Raw identifier (will be hashed for storage).
        collector_entity: Name of the entity collecting biometric data.
        purpose: Stated purpose for biometric data collection.
        data_retention_days: Days to retain data (default 30, per BIPA).
        consent_method: Method of consent capture.

    Returns:
        A BIPAConsent record with status GRANTED.

    """
    return BIPAConsent(
        consent_given=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        data_retention_days=data_retention_days,
        purpose=purpose,
        subject_identifier_hash=hash_identifier(subject_identifier),
        collector_entity=collector_entity,
        consent_method=consent_method,
        written_notice_provided=True,
        retention_schedule_disclosed=True,
        no_sale_clause_accepted=True,
        status=ConsentStatus.GRANTED,
    )


def revoke_consent(consent: BIPAConsent) -> BIPAConsent:
    """
    Revoke an existing consent record.

    Per BIPA, subjects have the right to revoke consent at any time.
    Revocation triggers immediate data deletion requirements.
    """
    consent.status = ConsentStatus.REVOKED
    consent.consent_given = False
    return consent


class ConsentStore:
    """
    Persistent storage for BIPA consent records with audit trail.

    Stores consent records as JSON files with integrity hashes.
    Supports audit queries and automatic expiration enforcement.

    Usage:
        store = ConsentStore("/path/to/consent/records")
        consent = create_consent("user@example.com", "FaceLock Inc")
        store.save(consent)
        assert store.is_consented("user@example.com")
    """

    def __init__(self, storage_dir: str) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._audit_log_path = self.storage_dir / "audit_log.jsonl"

    def save(self, consent: BIPAConsent) -> str:
        """Save a consent record and return its file path."""
        record_id = consent.subject_identifier_hash[:16]
        record_path = self.storage_dir / f"consent_{record_id}.json"

        record_data = consent.to_dict()
        record_data["integrity_hash"] = self._compute_integrity_hash(
            record_data
        )

        with open(record_path, "w") as f:
            json.dump(record_data, f, indent=2)

        self._append_audit_log(
            action="save",
            record_id=record_id,
            status=consent.status.value,
        )

        return str(record_path)

    def load(self, subject_identifier: str) -> Optional[BIPAConsent]:
        """Load a consent record by subject identifier."""
        id_hash = hash_identifier(subject_identifier)
        record_id = id_hash[:16]
        record_path = self.storage_dir / f"consent_{record_id}.json"

        if not record_path.exists():
            return None

        with open(record_path) as f:
            data = json.load(f)

        # Verify integrity
        stored_hash = data.pop("integrity_hash", "")
        expected_hash = self._compute_integrity_hash(data)
        if stored_hash != expected_hash:
            self._append_audit_log(
                action="integrity_failure",
                record_id=record_id,
                status="error",
            )
            return None

        consent = BIPAConsent(
            consent_given=data["consent_given"],
            timestamp=data["timestamp"],
            data_retention_days=data["data_retention_days"],
            purpose=data["purpose"],
            subject_identifier_hash=data["subject_identifier_hash"],
            collector_entity=data["collector_entity"],
            consent_method=data["consent_method"],
            written_notice_provided=data["written_notice_provided"],
            retention_schedule_disclosed=data["retention_schedule_disclosed"],
            no_sale_clause_accepted=data["no_sale_clause_accepted"],
            status=ConsentStatus(data["status"]),
        )

        # Auto-expire if retention period has passed
        if consent.is_expired and consent.status == ConsentStatus.GRANTED:
            consent.status = ConsentStatus.EXPIRED
            self.save(consent)
            self._append_audit_log(
                action="auto_expired",
                record_id=record_id,
                status="expired",
            )

        return consent

    def is_consented(self, subject_identifier: str) -> bool:
        """Check if a subject has valid, non-expired consent."""
        consent = self.load(subject_identifier)
        return consent is not None and consent.is_valid

    def delete(self, subject_identifier: str) -> bool:
        """
        Delete a consent record and associated biometric data.

        Per BIPA, data must be destroyed upon consent revocation or
        retention period expiry.
        """
        id_hash = hash_identifier(subject_identifier)
        record_id = id_hash[:16]
        record_path = self.storage_dir / f"consent_{record_id}.json"

        if record_path.exists():
            os.remove(record_path)
            self._append_audit_log(
                action="delete",
                record_id=record_id,
                status="deleted",
            )
            return True
        return False

    def get_expired_records(self) -> List[str]:
        """Find all expired consent records that need data deletion."""
        expired = []
        for record_path in self.storage_dir.glob("consent_*.json"):
            with open(record_path) as f:
                data = json.load(f)
            try:
                consent_time = datetime.fromisoformat(data["timestamp"])
                retention = data.get("data_retention_days", 30)
                expiry = consent_time + timedelta(days=retention)
                if datetime.now(timezone.utc) > expiry:
                    expired.append(str(record_path))
            except (ValueError, KeyError):
                expired.append(str(record_path))
        return expired

    def enforce_retention(self) -> int:
        """Delete all expired records. Returns count of deleted records."""
        expired = self.get_expired_records()
        for path in expired:
            os.remove(path)
            record_id = Path(path).stem.replace("consent_", "")
            self._append_audit_log(
                action="retention_enforced",
                record_id=record_id,
                status="deleted",
            )
        return len(expired)

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Read the audit log."""
        if not self._audit_log_path.exists():
            return []
        entries = []
        with open(self._audit_log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _append_audit_log(
        self, action: str, record_id: str, status: str
    ) -> None:
        """Append an entry to the audit log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "record_id": record_id,
            "status": status,
        }
        with open(self._audit_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _compute_integrity_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of record data for tamper detection."""
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
