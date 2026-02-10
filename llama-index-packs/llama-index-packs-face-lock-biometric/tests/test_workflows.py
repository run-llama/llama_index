"""Tests for the dual-path workflow system."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from llama_index.packs.face_lock_biometric.face_lock import (
    EyeShape,
    FacialMeasurements,
    FitzpatrickScale,
)
from llama_index.packs.face_lock_biometric.workflows import (
    CharacterProfile,
    RealFaceWorkflow,
    SyntheticWorkflow,
    WorkflowMode,
    generate_synthetic_measurements,
)


class TestGenerateSyntheticMeasurements:
    """Test synthetic measurement generation."""

    def test_returns_facial_measurements(self) -> None:
        m = generate_synthetic_measurements()
        assert isinstance(m, FacialMeasurements)

    def test_deterministic_with_seed(self) -> None:
        m1 = generate_synthetic_measurements(seed=42)
        m2 = generate_synthetic_measurements(seed=42)
        assert m1.gonial_angle == m2.gonial_angle
        assert m1.facial_index == m2.facial_index
        assert m1.canthal_tilt == m2.canthal_tilt

    def test_different_seeds_different_results(self) -> None:
        m1 = generate_synthetic_measurements(seed=1)
        m2 = generate_synthetic_measurements(seed=2)
        # At least one measurement should differ
        assert (
            m1.gonial_angle != m2.gonial_angle
            or m1.facial_index != m2.facial_index
        )

    def test_measurements_in_biological_range(self) -> None:
        for seed in range(50):
            m = generate_synthetic_measurements(seed=seed)
            assert 118.0 <= m.gonial_angle <= 140.0
            assert 1.15 <= m.facial_index <= 1.50
            assert -6.0 <= m.canthal_tilt <= 8.0
            assert 85.0 <= m.nasal_tip_rotation <= 115.0
            assert 0.08 <= m.zygomatic_prominence <= 0.25

    def test_overrides(self) -> None:
        m = generate_synthetic_measurements(
            seed=42, overrides={"gonial_angle": 125.0}
        )
        assert m.gonial_angle == 125.0

    def test_eye_shape_is_valid_enum(self) -> None:
        m = generate_synthetic_measurements(seed=42)
        assert isinstance(m.eye_shape, EyeShape)

    def test_fitzpatrick_is_valid_enum(self) -> None:
        m = generate_synthetic_measurements(seed=42)
        assert isinstance(m.fitzpatrick_type, FitzpatrickScale)


class TestSyntheticWorkflow:
    """Test the BIPA-safe synthetic workflow."""

    def test_create_character(self) -> None:
        wf = SyntheticWorkflow()
        profile = wf.create_character(name="Hero_001", seed=42)
        assert isinstance(profile, CharacterProfile)
        assert profile.name == "Hero_001"
        assert profile.workflow_mode == WorkflowMode.SYNTHETIC
        assert profile.consent is None

    def test_create_character_with_overrides(self) -> None:
        wf = SyntheticWorkflow()
        profile = wf.create_character(
            name="Test",
            overrides={"gonial_angle": 130.0},
        )
        assert profile.measurements.gonial_angle == 130.0

    def test_create_character_with_tags(self) -> None:
        wf = SyntheticWorkflow()
        profile = wf.create_character(
            name="Villain",
            tags=["antagonist", "sci-fi"],
            notes="Sharp features",
        )
        assert profile.tags == ["antagonist", "sci-fi"]
        assert profile.notes == "Sharp features"

    def test_create_batch(self) -> None:
        wf = SyntheticWorkflow()
        batch = wf.create_batch("NPC", count=5, base_seed=100)
        assert len(batch) == 5
        assert batch[0].name == "NPC_001"
        assert batch[4].name == "NPC_005"
        # Each should have different measurements
        angles = [c.measurements.gonial_angle for c in batch]
        assert len(set(angles)) > 1  # Not all the same

    def test_mode_is_synthetic(self) -> None:
        wf = SyntheticWorkflow()
        assert wf.mode == WorkflowMode.SYNTHETIC


class TestCharacterProfile:
    """Test CharacterProfile serialization."""

    def test_to_dict_synthetic(self) -> None:
        m = generate_synthetic_measurements(seed=42)
        profile = CharacterProfile(
            name="Test",
            measurements=m,
            workflow_mode=WorkflowMode.SYNTHETIC,
            tags=["test"],
        )
        d = profile.to_dict()
        assert d["name"] == "Test"
        assert d["workflow_mode"] == "synthetic"
        assert "measurements" in d
        assert d["tags"] == ["test"]
        assert "consent" not in d

    def test_to_dict_with_consent(self) -> None:
        from llama_index.packs.face_lock_biometric.bipa_consent import (
            create_consent,
        )

        m = generate_synthetic_measurements(seed=42)
        consent = create_consent("user@test.com", "TestCorp")
        profile = CharacterProfile(
            name="Actor",
            measurements=m,
            workflow_mode=WorkflowMode.REAL_FACE,
            consent=consent,
        )
        d = profile.to_dict()
        assert d["workflow_mode"] == "real_face"
        assert "consent" in d
        assert d["consent"]["consent_given"] is True


class TestRealFaceWorkflow:
    """Test the BIPA-regulated real face workflow."""

    def test_requires_consent_before_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = RealFaceWorkflow(consent_storage_dir=tmpdir)
            with pytest.raises(PermissionError, match="BIPA consent"):
                wf.analyze_face(
                    image_input="test.jpg",
                    subject_identifier="user@test.com",
                    character_name="Test",
                )

    def test_consent_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = RealFaceWorkflow(consent_storage_dir=tmpdir)
            consent = wf.request_consent("user@test.com")
            assert consent.is_valid
            assert wf.consent_store.is_consented("user@test.com")

    def test_mode_is_real_face(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wf = RealFaceWorkflow(consent_storage_dir=tmpdir)
            assert wf.mode == WorkflowMode.REAL_FACE


class TestWorkflowMode:
    """Test WorkflowMode enum."""

    def test_values(self) -> None:
        assert WorkflowMode.SYNTHETIC.value == "synthetic"
        assert WorkflowMode.REAL_FACE.value == "real_face"
