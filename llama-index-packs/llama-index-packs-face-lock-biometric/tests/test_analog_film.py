"""Tests for the analog film processing pipeline."""

import numpy as np
import pytest

from llama_index.packs.face_lock_biometric.analog_film import (
    AnalogFilmProcessor,
    FilmPreset,
    FilmStockProfile,
    FILM_PRESETS,
)


def _make_test_image(h: int = 100, w: int = 150) -> np.ndarray:
    """Create a test BGR image with gradient."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Horizontal gradient
    for x in range(w):
        val = int(255 * x / w)
        img[:, x, :] = val
    return img


def _make_bright_image(h: int = 100, w: int = 150) -> np.ndarray:
    """Create a bright test image (for halation testing)."""
    img = np.full((h, w, 3), 220, dtype=np.uint8)
    return img


class TestFilmStockProfile:
    """Test FilmStockProfile defaults."""

    def test_default_profile(self) -> None:
        p = FilmStockProfile()
        assert p.name == "Custom"
        assert p.grain_intensity == 0.03
        assert p.halation_strength == 0.0
        assert p.gate_weave_amplitude == 0.0

    def test_presets_exist(self) -> None:
        assert FilmPreset.AGFA_CNS2_EXPIRED_1972 in FILM_PRESETS
        assert FilmPreset.KODACHROME_25 in FILM_PRESETS
        assert FilmPreset.KODAK_VISION3_500T_PUSH2 in FILM_PRESETS

    def test_preset_names(self) -> None:
        agfa = FILM_PRESETS[FilmPreset.AGFA_CNS2_EXPIRED_1972]
        assert "Agfa" in agfa.name
        kodachrome = FILM_PRESETS[FilmPreset.KODACHROME_25]
        assert "Kodachrome" in kodachrome.name


class TestAnalogFilmProcessor:
    """Test the film processing pipeline."""

    def test_process_returns_same_shape(self) -> None:
        img = _make_test_image()
        proc = AnalogFilmProcessor(preset=FilmPreset.KODACHROME_25)
        result = proc.process(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_process_modifies_image(self) -> None:
        img = _make_test_image()
        proc = AnalogFilmProcessor(preset=FilmPreset.AGFA_CNS2_EXPIRED_1972)
        result = proc.process(img)
        # The result should be different from the input
        assert not np.array_equal(result, img)

    def test_process_stays_in_valid_range(self) -> None:
        img = _make_test_image()
        proc = AnalogFilmProcessor(preset=FilmPreset.KODAK_VISION3_500T_PUSH2)
        result = proc.process(img)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_custom_profile(self) -> None:
        profile = FilmStockProfile(
            grain_intensity=0.1,
            halation_strength=0.2,
            vignette_strength=0.5,
        )
        proc = AnalogFilmProcessor(profile=profile)
        img = _make_test_image()
        result = proc.process(img)
        assert result.shape == img.shape

    def test_no_effects_profile(self) -> None:
        profile = FilmStockProfile(
            grain_intensity=0.0,
            halation_strength=0.0,
            chromatic_aberration_px=0.0,
            gate_weave_amplitude=0.0,
            vignette_strength=0.0,
            shoulder_compression=0.0,
            contrast_adjustment=0.0,
            black_point_lift=0.0,
            midtone_saturation=1.0,
        )
        proc = AnalogFilmProcessor(profile=profile)
        img = _make_test_image()
        result = proc.process(img)
        # With no effects, should be very similar (some floating point diff)
        diff = np.abs(result.astype(float) - img.astype(float))
        # Allow small floating point conversion differences
        assert diff.mean() < 2.0

    def test_custom_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="custom"):
            AnalogFilmProcessor(preset=FilmPreset.CUSTOM)

    def test_process_batch(self) -> None:
        proc = AnalogFilmProcessor(preset=FilmPreset.KODACHROME_25)
        images = [_make_test_image() for _ in range(3)]
        results = proc.process_batch(images)
        assert len(results) == 3
        for r in results:
            assert r.shape == images[0].shape

    def test_grain_varies_per_call(self) -> None:
        proc = AnalogFilmProcessor(
            profile=FilmStockProfile(
                grain_intensity=0.2,
                halation_strength=0.0,
                chromatic_aberration_px=0.0,
                gate_weave_amplitude=0.0,
                vignette_strength=0.0,
                shoulder_compression=0.0,
                contrast_adjustment=0.0,
                black_point_lift=0.0,
                midtone_saturation=1.0,
            )
        )
        img = _make_test_image()
        r1 = proc.process(img)
        r2 = proc.process(img)
        # Grain is random, so results should differ
        assert not np.array_equal(r1, r2)

    def test_vignette_darkens_edges(self) -> None:
        profile = FilmStockProfile(
            grain_intensity=0.0,
            vignette_strength=0.8,
            halation_strength=0.0,
            chromatic_aberration_px=0.0,
            gate_weave_amplitude=0.0,
            shoulder_compression=0.0,
            contrast_adjustment=0.0,
            black_point_lift=0.0,
            midtone_saturation=1.0,
        )
        proc = AnalogFilmProcessor(profile=profile)
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = proc.process(img)

        # Center should be brighter than corners
        center_val = float(result[50, 50].mean())
        corner_val = float(result[0, 0].mean())
        assert center_val > corner_val

    def test_black_point_lift_raises_shadows(self) -> None:
        profile = FilmStockProfile(
            grain_intensity=0.0,
            black_point_lift=0.2,
            halation_strength=0.0,
            chromatic_aberration_px=0.0,
            gate_weave_amplitude=0.0,
            vignette_strength=0.0,
            shoulder_compression=0.0,
            contrast_adjustment=0.0,
            midtone_saturation=1.0,
        )
        proc = AnalogFilmProcessor(profile=profile)
        img = np.zeros((50, 50, 3), dtype=np.uint8)  # Pure black
        result = proc.process(img)
        # Black point lift should make pure black into dark gray
        assert result.mean() > 10
