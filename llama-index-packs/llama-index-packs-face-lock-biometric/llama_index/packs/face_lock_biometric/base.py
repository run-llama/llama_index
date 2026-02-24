"""
Face Lock Biometric Pack — LlamaIndex BaseLlamaPack integration.

Provides the FaceLockBiometricPack class that integrates Face Lock's
biometric measurement engine, prompt generation, and drift detection
into the LlamaIndex pack ecosystem.
"""

from typing import Any, Dict, List, Optional

from llama_index.core.llama_pack.base import BaseLlamaPack

from .analog_film import AnalogFilmProcessor, FilmPreset
from .face_lock import DriftDetector, FaceAnalyzer, FacialMeasurements
from .prompt_generators import Platform, PromptGenerator
from .workflows import (
    CharacterProfile,
    RealFaceWorkflow,
    SyntheticWorkflow,
    WorkflowMode,
)


class FaceLockBiometricPack(BaseLlamaPack):
    """
    LlamaIndex pack for biometric character consistency in AI image generation.

    Extracts precise facial measurements from reference images (or generates
    synthetic ones), converts them to structured prompts for AI image
    generators, and detects drift across generated batches.

    Usage (Synthetic — BIPA-safe):
        pack = FaceLockBiometricPack(mode="synthetic")
        result = pack.run(
            character_name="Hero_001",
            seed=42,
            platform="reve",
        )
        print(result["positive_prompt"])

    Usage (Real Face — requires consent):
        pack = FaceLockBiometricPack(
            mode="real_face",
            image_path="reference.jpg",
            subject_identifier="actor@example.com",
            collector_entity="MyStudio",
        )
        result = pack.run(
            character_name="Lead_Actor",
            platform="flux",
        )
    """

    def __init__(
        self,
        mode: str = "synthetic",
        image_path: Optional[str] = None,
        subject_identifier: Optional[str] = None,
        collector_entity: str = "FaceLock",
        consent_storage_dir: str = ".face_lock_consent",
        platform: str = "generic",
        include_technical: bool = False,
        film_preset: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Face Lock pack.

        Args:
            mode: "synthetic" or "real_face".
            image_path: Path to reference image (real_face mode only).
            subject_identifier: Subject identifier for BIPA consent
                (real_face mode only).
            collector_entity: Entity collecting biometric data.
            consent_storage_dir: Directory for consent records.
            platform: Target AI platform ("reve", "flux", "midjourney", "generic").
            include_technical: Include raw biometric values in prompts.
            film_preset: Optional film stock preset name.

        """
        self.mode = WorkflowMode(mode)
        self.image_path = image_path
        self.subject_identifier = subject_identifier
        self.platform = Platform(platform)
        self.include_technical = include_technical

        # Initialize components
        self._analyzer: Optional[FaceAnalyzer] = None
        self._prompt_gen = PromptGenerator(
            platform=self.platform,
            include_technical=self.include_technical,
        )

        # Film processing
        self._film_processor: Optional[AnalogFilmProcessor] = None
        if film_preset:
            preset_map = {
                "agfa_cns2_expired_1972": FilmPreset.AGFA_CNS2_EXPIRED_1972,
                "kodachrome_25": FilmPreset.KODACHROME_25,
                "kodak_vision3_500t_push2": FilmPreset.KODAK_VISION3_500T_PUSH2,
            }
            if film_preset in preset_map:
                self._film_processor = AnalogFilmProcessor(
                    preset=preset_map[film_preset]
                )

        # Workflow setup
        if self.mode == WorkflowMode.SYNTHETIC:
            self._workflow: Any = SyntheticWorkflow()
        else:
            self._real_face_workflow = RealFaceWorkflow(
                consent_storage_dir=consent_storage_dir,
                collector_entity=collector_entity,
            )
            self._workflow = self._real_face_workflow

            # Auto-request consent if subject provided
            if subject_identifier:
                self._real_face_workflow.request_consent(subject_identifier)

        # Store current measurements
        self._measurements: Optional[FacialMeasurements] = None
        self._profile: Optional[CharacterProfile] = None

    def get_modules(self) -> Dict[str, Any]:
        """Get all pack modules."""
        modules: Dict[str, Any] = {
            "mode": self.mode.value,
            "platform": self.platform.value,
            "prompt_generator": self._prompt_gen,
            "workflow": self._workflow,
        }
        if self._analyzer is not None:
            modules["analyzer"] = self._analyzer
        if self._film_processor is not None:
            modules["film_processor"] = self._film_processor
        if self._measurements is not None:
            modules["measurements"] = self._measurements
        if self._profile is not None:
            modules["profile"] = self._profile
        return modules

    def run(
        self,
        character_name: str = "Character_001",
        seed: Optional[int] = None,
        overrides: Optional[Dict[str, float]] = None,
        platform: Optional[str] = None,
        additional_context: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run the Face Lock pipeline.

        For synthetic mode: generates random measurements and prompts.
        For real_face mode: analyzes the reference image and generates prompts.

        Args:
            character_name: Name for the character profile.
            seed: Random seed (synthetic mode).
            overrides: Measurement overrides (synthetic mode).
            platform: Override the platform for this run.
            additional_context: Extra context for prompt generation.

        Returns:
            Dict with measurements, prompts, and profile data.

        """
        # Override platform if specified
        prompt_gen = self._prompt_gen
        if platform:
            prompt_gen = PromptGenerator(
                platform=Platform(platform),
                include_technical=self.include_technical,
            )

        # Generate or extract measurements
        if self.mode == WorkflowMode.SYNTHETIC:
            profile = self._workflow.create_character(
                name=character_name,
                seed=seed,
                overrides=overrides,
            )
        else:
            if self.image_path is None:
                raise ValueError(
                    "image_path is required for real_face mode."
                )
            if self.subject_identifier is None:
                raise ValueError(
                    "subject_identifier is required for real_face mode."
                )
            profile = self._workflow.analyze_face(
                image_input=self.image_path,
                subject_identifier=self.subject_identifier,
                character_name=character_name,
            )

        self._profile = profile
        self._measurements = profile.measurements

        # Generate prompts
        positive, negative = prompt_gen.generate(
            profile.measurements,
            additional_context=additional_context,
        )

        return {
            "character_name": character_name,
            "workflow_mode": self.mode.value,
            "measurements": profile.measurements.to_dict(),
            "positive_prompt": positive,
            "negative_prompt": negative,
            "profile": profile.to_dict(),
        }

    def check_drift(
        self,
        generated_images: List[Any],
        tolerances: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check generated images for drift from locked measurements.

        Args:
            generated_images: List of image paths or arrays to check.
            tolerances: Optional custom tolerances per measurement.

        Returns:
            List of drift report dicts.

        """
        if self._measurements is None:
            raise ValueError(
                "No measurements locked. Call run() first."
            )

        if self._analyzer is None:
            self._analyzer = FaceAnalyzer()

        detector = DriftDetector(
            target=self._measurements,
            tolerances=tolerances,
            analyzer=self._analyzer,
        )
        reports = detector.check_batch(generated_images)

        return [
            {
                "image_index": r.image_index,
                "overall_drift_score": r.overall_drift_score,
                "is_within_tolerance": r.is_within_tolerance,
                "deviations": r.deviations,
                "details": r.details,
            }
            for r in reports
        ]

    def apply_film(
        self,
        image: Any,
        verify_measurements: bool = True,
    ) -> Dict[str, Any]:
        """
        Apply analog film processing to an image.

        Args:
            image: Image path or numpy array.
            verify_measurements: If True, verify face lock holds after processing.

        Returns:
            Dict with processed image and optional drift report.

        """
        import cv2

        if self._film_processor is None:
            raise ValueError(
                "No film preset configured. Initialize with film_preset= parameter."
            )

        # Load image if path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image

        from .analog_film import process_with_verification

        analyzer = self._analyzer if verify_measurements else None
        target = self._measurements if verify_measurements else None

        processed, drift_report = process_with_verification(
            image=img,
            processor=self._film_processor,
            target_measurements=target,
            analyzer=analyzer,
        )

        result: Dict[str, Any] = {"processed_image": processed}
        if drift_report is not None:
            result["drift_report"] = {
                "overall_drift_score": drift_report.overall_drift_score,
                "is_within_tolerance": drift_report.is_within_tolerance,
                "details": drift_report.details,
            }

        return result

    def close(self) -> None:
        """Release all resources."""
        if self._analyzer is not None:
            self._analyzer.close()
        if hasattr(self._workflow, "close"):
            self._workflow.close()
