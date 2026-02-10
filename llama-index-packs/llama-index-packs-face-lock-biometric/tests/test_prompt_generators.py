"""Tests for the prompt generation system."""

import pytest

from llama_index.packs.face_lock_biometric.face_lock import (
    EyeShape,
    FacialMeasurements,
    FitzpatrickScale,
)
from llama_index.packs.face_lock_biometric.prompt_generators import (
    Platform,
    PromptGenerator,
    generate_prompt_for_platform,
)


def _make_measurements() -> FacialMeasurements:
    """Create test measurements with known values."""
    return FacialMeasurements(
        gonial_angle=128.0,
        facial_index=1.30,
        canthal_tilt=2.0,
        nasal_tip_rotation=97.0,
        zygomatic_prominence=0.15,
        eye_shape=EyeShape.ALMOND,
        fitzpatrick_type=FitzpatrickScale.TYPE_III,
        philtrum_length=0.25,
        lip_fullness_ratio=0.55,
        brow_arch_height=0.05,
        face_symmetry_score=0.92,
    )


class TestPromptGenerator:
    """Test PromptGenerator class."""

    def test_generic_prompt(self) -> None:
        gen = PromptGenerator(platform=Platform.GENERIC)
        positive, negative = gen.generate(_make_measurements())
        assert isinstance(positive, str)
        assert isinstance(negative, str)
        assert len(positive) > 20
        assert len(negative) > 10

    def test_reve_prompt_format(self) -> None:
        gen = PromptGenerator(platform=Platform.REVE)
        positive, negative = gen.generate(_make_measurements())
        assert "photorealistic portrait" in positive
        assert "8k" in positive

    def test_flux_prompt_format(self) -> None:
        gen = PromptGenerator(platform=Platform.FLUX)
        positive, negative = gen.generate(_make_measurements())
        assert "photo" in positive.lower()
        assert "sharp focus" in positive

    def test_midjourney_prompt_format(self) -> None:
        gen = PromptGenerator(platform=Platform.MIDJOURNEY)
        positive, negative = gen.generate(_make_measurements())
        assert "--v 6" in positive
        assert "--style raw" in positive
        assert "--no" in negative

    def test_includes_jawline_descriptor(self) -> None:
        gen = PromptGenerator(platform=Platform.GENERIC)

        # Sharp jaw
        m = _make_measurements()
        m.gonial_angle = 122.0
        positive, _ = gen.generate(m)
        assert "sharp" in positive.lower()

        # Soft jaw
        m.gonial_angle = 133.0
        positive, _ = gen.generate(m)
        assert "soft" in positive.lower() or "rounded" in positive.lower()

    def test_includes_eye_shape(self) -> None:
        gen = PromptGenerator(platform=Platform.GENERIC)
        m = _make_measurements()
        m.eye_shape = EyeShape.HOODED
        positive, _ = gen.generate(m)
        assert "hooded" in positive.lower()

    def test_includes_skin_tone(self) -> None:
        gen = PromptGenerator(platform=Platform.GENERIC)
        m = _make_measurements()
        m.fitzpatrick_type = FitzpatrickScale.TYPE_V
        positive, _ = gen.generate(m)
        assert "brown skin" in positive.lower()

    def test_additional_context(self) -> None:
        gen = PromptGenerator(platform=Platform.GENERIC)
        positive, _ = gen.generate(
            _make_measurements(),
            additional_context="wearing a red jacket, rainy street",
        )
        assert "red jacket" in positive
        assert "rainy street" in positive

    def test_include_technical(self) -> None:
        gen = PromptGenerator(platform=Platform.GENERIC, include_technical=True)
        positive, _ = gen.generate(_make_measurements())
        assert "BIOMETRICS" in positive
        assert "jaw=128.0" in positive

    def test_no_technical_by_default(self) -> None:
        gen = PromptGenerator(platform=Platform.GENERIC)
        positive, _ = gen.generate(_make_measurements())
        assert "BIOMETRICS" not in positive

    def test_style_prefix_suffix(self) -> None:
        gen = PromptGenerator(
            platform=Platform.GENERIC,
            style_prefix="cinematic film still",
            style_suffix="directed by Kubrick",
        )
        positive, _ = gen.generate(_make_measurements())
        assert positive.startswith("cinematic film still")
        assert "directed by Kubrick" in positive

    def test_negative_prompt_sharp_jaw_prevents_rounding(self) -> None:
        gen = PromptGenerator(platform=Platform.GENERIC)
        m = _make_measurements()
        m.gonial_angle = 122.0  # Sharp
        _, negative = gen.generate(m)
        assert "round jaw" in negative.lower()

    def test_negative_prompt_round_jaw_prevents_sharpening(self) -> None:
        gen = PromptGenerator(platform=Platform.GENERIC)
        m = _make_measurements()
        m.gonial_angle = 136.0  # Round
        _, negative = gen.generate(m)
        assert "sharp jaw" in negative.lower()

    def test_format_for_api(self) -> None:
        gen = PromptGenerator(platform=Platform.REVE)
        result = gen.format_for_api(_make_measurements())
        assert "positive_prompt" in result
        assert "negative_prompt" in result
        assert result["platform"] == "reve"

    def test_character_prompt(self) -> None:
        gen = PromptGenerator(platform=Platform.MIDJOURNEY)
        positive, negative = gen.generate_character_prompt(
            _make_measurements(),
            character_name="Aria",
            scene_description="standing in a forest",
            camera_angle="close-up portrait",
            lighting="golden hour",
        )
        assert '"Aria"' in positive
        assert "forest" in positive
        assert "close-up" in positive


class TestConvenienceFunction:
    """Test generate_prompt_for_platform convenience function."""

    def test_basic(self) -> None:
        positive, negative = generate_prompt_for_platform(
            _make_measurements(), platform=Platform.FLUX
        )
        assert isinstance(positive, str)
        assert isinstance(negative, str)
        assert "photo" in positive.lower()

    def test_with_context(self) -> None:
        positive, _ = generate_prompt_for_platform(
            _make_measurements(),
            platform=Platform.GENERIC,
            context="on a spaceship",
        )
        assert "spaceship" in positive

    def test_with_technical(self) -> None:
        positive, _ = generate_prompt_for_platform(
            _make_measurements(),
            include_technical=True,
        )
        assert "BIOMETRICS" in positive
