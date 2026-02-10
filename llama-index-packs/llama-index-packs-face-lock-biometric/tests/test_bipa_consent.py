"""Tests for the BIPA consent management layer."""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from llama_index.packs.face_lock_biometric.bipa_consent import (
    BIPAConsent,
    ConsentStatus,
    ConsentStore,
    create_consent,
    hash_identifier,
    revoke_consent,
)


class TestHashIdentifier:
    """Test identifier hashing."""

    def test_deterministic(self) -> None:
        h1 = hash_identifier("test@example.com")
        h2 = hash_identifier("test@example.com")
        assert h1 == h2

    def test_different_inputs_different_hashes(self) -> None:
        h1 = hash_identifier("user1@example.com")
        h2 = hash_identifier("user2@example.com")
        assert h1 != h2

    def test_returns_hex_string(self) -> None:
        h = hash_identifier("test")
        assert len(h) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in h)


class TestBIPAConsent:
    """Test BIPAConsent dataclass."""

    def test_default_values(self) -> None:
        c = BIPAConsent()
        assert not c.consent_given
        assert c.data_retention_days == 30
        assert c.purpose == "AI character consistency"
        assert c.status == ConsentStatus.PENDING

    def test_auto_timestamp(self) -> None:
        c = BIPAConsent()
        assert c.timestamp  # Should be auto-filled
        dt = datetime.fromisoformat(c.timestamp)
        assert dt.tzinfo is not None  # Should be timezone-aware

    def test_is_valid_requires_all_disclosures(self) -> None:
        c = BIPAConsent(
            consent_given=True,
            status=ConsentStatus.GRANTED,
            written_notice_provided=True,
            retention_schedule_disclosed=True,
            no_sale_clause_accepted=False,  # Missing
        )
        assert not c.is_valid

    def test_is_valid_when_complete(self) -> None:
        c = create_consent("test@example.com", "TestCorp")
        assert c.is_valid

    def test_is_expired_old_consent(self) -> None:
        old_time = (
            datetime.now(timezone.utc) - timedelta(days=31)
        ).isoformat()
        c = BIPAConsent(
            consent_given=True,
            timestamp=old_time,
            data_retention_days=30,
            status=ConsentStatus.GRANTED,
            written_notice_provided=True,
            retention_schedule_disclosed=True,
            no_sale_clause_accepted=True,
        )
        assert c.is_expired
        assert not c.is_valid

    def test_is_not_expired_recent(self) -> None:
        c = create_consent("test@example.com", "TestCorp")
        assert not c.is_expired

    def test_expiry_date(self) -> None:
        c = create_consent("test@example.com", "TestCorp")
        expiry = c.expiry_date
        assert expiry  # Should be a valid ISO string
        dt = datetime.fromisoformat(expiry)
        assert dt > datetime.now(timezone.utc)

    def test_to_dict(self) -> None:
        c = create_consent("test@example.com", "TestCorp")
        d = c.to_dict()
        assert d["consent_given"] is True
        assert d["status"] == "granted"
        assert d["collector_entity"] == "TestCorp"
        assert d["no_sale_clause_accepted"] is True
        assert "expiry_date" in d


class TestCreateConsent:
    """Test consent creation helper."""

    def test_creates_valid_consent(self) -> None:
        c = create_consent(
            subject_identifier="user@example.com",
            collector_entity="TestCorp",
        )
        assert c.consent_given
        assert c.status == ConsentStatus.GRANTED
        assert c.written_notice_provided
        assert c.retention_schedule_disclosed
        assert c.no_sale_clause_accepted
        assert c.collector_entity == "TestCorp"

    def test_hashes_identifier(self) -> None:
        c = create_consent("user@example.com", "TestCorp")
        assert c.subject_identifier_hash == hash_identifier("user@example.com")

    def test_custom_retention(self) -> None:
        c = create_consent(
            "user@example.com", "TestCorp", data_retention_days=7
        )
        assert c.data_retention_days == 7


class TestRevokeConsent:
    """Test consent revocation."""

    def test_revokes_consent(self) -> None:
        c = create_consent("user@example.com", "TestCorp")
        assert c.is_valid
        revoke_consent(c)
        assert not c.is_valid
        assert c.status == ConsentStatus.REVOKED
        assert not c.consent_given


class TestConsentStore:
    """Test persistent consent storage."""

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConsentStore(tmpdir)
            consent = create_consent("user@example.com", "TestCorp")
            store.save(consent)

            loaded = store.load("user@example.com")
            assert loaded is not None
            assert loaded.consent_given
            assert loaded.status == ConsentStatus.GRANTED

    def test_is_consented(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConsentStore(tmpdir)
            assert not store.is_consented("user@example.com")

            consent = create_consent("user@example.com", "TestCorp")
            store.save(consent)
            assert store.is_consented("user@example.com")

    def test_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConsentStore(tmpdir)
            consent = create_consent("user@example.com", "TestCorp")
            store.save(consent)
            assert store.is_consented("user@example.com")

            store.delete("user@example.com")
            assert not store.is_consented("user@example.com")

    def test_delete_nonexistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConsentStore(tmpdir)
            assert not store.delete("nonexistent@example.com")

    def test_load_nonexistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConsentStore(tmpdir)
            assert store.load("nonexistent@example.com") is None

    def test_audit_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConsentStore(tmpdir)
            consent = create_consent("user@example.com", "TestCorp")
            store.save(consent)
            store.load("user@example.com")

            log = store.get_audit_log()
            assert len(log) >= 1
            assert log[0]["action"] == "save"

    def test_integrity_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConsentStore(tmpdir)
            consent = create_consent("user@example.com", "TestCorp")
            path = store.save(consent)

            # Tamper with the file
            with open(path) as f:
                data = json.load(f)
            data["consent_given"] = False  # Tamper
            with open(path, "w") as f:
                json.dump(data, f)

            # Should fail integrity check
            loaded = store.load("user@example.com")
            assert loaded is None

    def test_enforce_retention_on_fresh_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConsentStore(tmpdir)
            assert store.enforce_retention() == 0
