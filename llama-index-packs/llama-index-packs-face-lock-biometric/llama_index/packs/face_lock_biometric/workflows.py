"""
Dual-path workflow system for Face Lock.

PATH A: SYNTHETIC (BIPA-Safe)
- Generate random character measurements from scratch
- No photo uploads = no biometric privacy laws apply
- Unlimited data retention
- Use case: Original characters, concept art

PATH B: REAL FACE (BIPA-Regulated)
- Upload reference photos of real people
- Requires explicit BIPA consent
- 30-day data retention limit
- Use case: Digital doubles, actor likeness preservation
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .bipa_consent import BIPAConsent, ConsentStore, create_consent
from .face_lock import (
    DriftDetector,
    EyeShape,
    FaceAnalyzer,
    FacialMeasurements,
    FitzpatrickScale,
)


class WorkflowMode(Enum):
    """Operating mode for Face Lock."""

    SYNTHETIC = "synthetic"  # Safe, default — no real faces
    REAL_FACE = "real_face"  # Requires BIPAConsent


@dataclass
class CharacterProfile:
    """A complete character identity with locked biometric parameters.

    Can be saved and loaded as a "Character Bible" entry for consistent
    generation across sessions.
    """

    name: str
    measurements: FacialMeasurements
    workflow_mode: WorkflowMode
    consent: Optional[BIPAConsent] = None
    tags: Optional[List[str]] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the profile for storage."""
        data: Dict[str, Any] = {
            "name": self.name,
            "measurements": self.measurements.to_dict(),
            "workflow_mode": self.workflow_mode.value,
            "tags": self.tags or [],
            "notes": self.notes,
        }
        if self.consent is not None:
            data["consent"] = self.consent.to_dict()
        return data


# Biologically plausible ranges for synthetic character generation
_SYNTHETIC_RANGES = {
    "gonial_angle": (118.0, 140.0),
    "facial_index": (1.15, 1.50),
    "canthal_tilt": (-6.0, 8.0),
    "nasal_tip_rotation": (85.0, 115.0),
    "zygomatic_prominence": (0.08, 0.25),
    "philtrum_length": (0.15, 0.35),
    "lip_fullness_ratio": (0.3, 0.9),
    "brow_arch_height": (0.02, 0.10),
    "face_symmetry_score": (0.85, 1.0),
}


def generate_synthetic_measurements(
    seed: Optional[int] = None,
    overrides: Optional[Dict[str, float]] = None,
) -> FacialMeasurements:
    """Generate biologically plausible random facial measurements.

    This is the BIPA-safe path — no real face data is involved.

    Args:
        seed: Random seed for reproducibility.
        overrides: Dict of measurement names to override with specific values.

    Returns:
        FacialMeasurements with randomly generated but plausible values.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    values = {}
    for name, (low, high) in _SYNTHETIC_RANGES.items():
        # Use truncated normal distribution centered in the range
        mean = (low + high) / 2
        std = (high - low) / 4  # 95% within range
        val = np.clip(np.random.normal(mean, std), low, high)
        values[name] = float(val)

    if overrides:
        values.update(overrides)

    eye_shapes = list(EyeShape)
    fitzpatrick_types = list(FitzpatrickScale)

    return FacialMeasurements(
        gonial_angle=values["gonial_angle"],
        facial_index=values["facial_index"],
        canthal_tilt=values["canthal_tilt"],
        nasal_tip_rotation=values["nasal_tip_rotation"],
        zygomatic_prominence=values["zygomatic_prominence"],
        eye_shape=random.choice(eye_shapes),
        fitzpatrick_type=random.choice(fitzpatrick_types),
        philtrum_length=values["philtrum_length"],
        lip_fullness_ratio=values["lip_fullness_ratio"],
        brow_arch_height=values["brow_arch_height"],
        face_symmetry_score=values["face_symmetry_score"],
    )


class SyntheticWorkflow:
    """BIPA-safe synthetic character generation workflow.

    Generates characters from scratch with random but biologically
    plausible measurements. No photo uploads, no biometric privacy
    concerns.
    """

    def __init__(self) -> None:
        self.mode = WorkflowMode.SYNTHETIC

    def create_character(
        self,
        name: str,
        seed: Optional[int] = None,
        overrides: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> CharacterProfile:
        """Generate a new synthetic character.

        Args:
            name: Character name/identifier.
            seed: Random seed for reproducibility.
            overrides: Override specific measurements.
            tags: Tags for organizing characters.
            notes: Free-text notes.

        Returns:
            CharacterProfile with synthetic measurements.
        """
        measurements = generate_synthetic_measurements(
            seed=seed, overrides=overrides
        )
        return CharacterProfile(
            name=name,
            measurements=measurements,
            workflow_mode=WorkflowMode.SYNTHETIC,
            tags=tags,
            notes=notes,
        )

    def create_batch(
        self,
        base_name: str,
        count: int,
        base_seed: Optional[int] = None,
    ) -> List[CharacterProfile]:
        """Generate a batch of unique synthetic characters."""
        characters = []
        for i in range(count):
            seed = (base_seed + i) if base_seed is not None else None
            char = self.create_character(
                name=f"{base_name}_{i + 1:03d}",
                seed=seed,
            )
            characters.append(char)
        return characters


class RealFaceWorkflow:
    """BIPA-regulated workflow for real face reference analysis.

    Requires explicit BIPA consent before any biometric data extraction.
    Enforces data retention limits and provides audit trails.
    """

    def __init__(
        self,
        consent_storage_dir: str = ".face_lock_consent",
        collector_entity: str = "FaceLock",
    ) -> None:
        self.mode = WorkflowMode.REAL_FACE
        self.consent_store = ConsentStore(consent_storage_dir)
        self.collector_entity = collector_entity
        self._analyzer: Optional[FaceAnalyzer] = None

    @property
    def analyzer(self) -> FaceAnalyzer:
        if self._analyzer is None:
            self._analyzer = FaceAnalyzer()
        return self._analyzer

    def request_consent(
        self,
        subject_identifier: str,
        purpose: str = "AI character consistency",
        data_retention_days: int = 30,
    ) -> BIPAConsent:
        """Create and store a BIPA consent record.

        This must be called BEFORE any biometric analysis.

        Args:
            subject_identifier: Email or name of the person whose face
                will be analyzed.
            purpose: Stated purpose of data collection.
            data_retention_days: How long data will be retained.

        Returns:
            BIPAConsent record (stored automatically).
        """
        consent = create_consent(
            subject_identifier=subject_identifier,
            collector_entity=self.collector_entity,
            purpose=purpose,
            data_retention_days=data_retention_days,
        )
        self.consent_store.save(consent)
        return consent

    def analyze_face(
        self,
        image_input: Any,
        subject_identifier: str,
        character_name: str,
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> CharacterProfile:
        """Analyze a real face image and create a character profile.

        Requires valid BIPA consent for the subject.

        Args:
            image_input: Image file path, numpy array, or PIL Image.
            subject_identifier: Identifier matching the consent record.
            character_name: Name for the character profile.
            tags: Optional tags.
            notes: Optional notes.

        Returns:
            CharacterProfile with extracted measurements.

        Raises:
            PermissionError: If no valid consent exists for the subject.
            ValueError: If no face is detected in the image.
        """
        if not self.consent_store.is_consented(subject_identifier):
            raise PermissionError(
                f"No valid BIPA consent found for subject "
                f"'{subject_identifier}'. Call request_consent() first. "
                f"BIPA violations carry $1,000-$5,000 penalties per incident."
            )

        consent = self.consent_store.load(subject_identifier)
        measurements = self.analyzer.analyze(image_input)

        return CharacterProfile(
            name=character_name,
            measurements=measurements,
            workflow_mode=WorkflowMode.REAL_FACE,
            consent=consent,
            tags=tags,
            notes=notes,
        )

    def check_drift(
        self,
        target: FacialMeasurements,
        generated_images: List[Any],
    ) -> List[Any]:
        """Check generated images for drift from target measurements."""
        detector = DriftDetector(target=target, analyzer=self.analyzer)
        return detector.check_batch(generated_images)

    def enforce_retention(self) -> int:
        """Enforce data retention policies. Returns count of purged records."""
        return self.consent_store.enforce_retention()

    def close(self) -> None:
        """Release resources."""
        if self._analyzer is not None:
            self._analyzer.close()

    def __enter__(self) -> "RealFaceWorkflow":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
