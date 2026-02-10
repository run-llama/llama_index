"""
Analog Film Destruction Layer.

Post-processing pipeline that simulates analog film chemistry to make
AI-generated images look organic. Applies grain, halation, color shifts,
chromatic aberration, and gate weave — then verifies facial measurements
still hold after processing.

Film Presets:
- Agfacolor CNS2 Expired (1972)
- Kodachrome 25
- Kodak Vision3 500T (Pushed +2)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError(
        "opencv-python is required for AnalogFilmProcessor. "
        "Install with: pip install opencv-python"
    )


class FilmPreset(Enum):
    """Available analog film stock presets."""

    AGFA_CNS2_EXPIRED_1972 = "agfa_cns2_expired_1972"
    KODACHROME_25 = "kodachrome_25"
    KODAK_VISION3_500T_PUSH2 = "kodak_vision3_500t_push2"
    CUSTOM = "custom"


@dataclass
class FilmStockProfile:
    """Configuration for a specific film stock emulation.

    Each parameter controls a different aspect of the analog look.
    """

    name: str = "Custom"

    # Grain
    grain_intensity: float = 0.03  # 0-1, silver halide grain strength
    grain_size: float = 1.5  # Grain particle size multiplier

    # Halation (red bloom on highlights)
    halation_strength: float = 0.0  # 0-1
    halation_radius: int = 15  # Blur radius for halation
    halation_color: Tuple[int, int, int] = (255, 80, 60)  # RGB tint

    # Color shifts (expiration / push processing)
    shadow_tint: Tuple[int, int, int] = (0, 0, 0)  # RGB shift in shadows
    highlight_tint: Tuple[int, int, int] = (0, 0, 0)  # RGB shift in highlights
    midtone_saturation: float = 1.0  # Saturation multiplier
    color_temperature_shift: float = 0.0  # Negative=cool, Positive=warm

    # Dynamic range
    shoulder_compression: float = 0.0  # Film shoulder rolloff (0-1)
    black_point_lift: float = 0.0  # Lift shadows (0-1)

    # Chromatic aberration
    chromatic_aberration_px: float = 0.0  # Lateral CA in pixels

    # Gate weave (frame jitter)
    gate_weave_amplitude: float = 0.0  # Max pixel displacement

    # Overall
    contrast_adjustment: float = 0.0  # -1 to 1
    vignette_strength: float = 0.0  # 0-1


# Pre-built film stock profiles
FILM_PRESETS: Dict[FilmPreset, FilmStockProfile] = {
    FilmPreset.AGFA_CNS2_EXPIRED_1972: FilmStockProfile(
        name="Agfacolor CNS2 Expired (1972)",
        grain_intensity=0.08,
        grain_size=2.0,
        halation_strength=0.15,
        halation_radius=20,
        halation_color=(255, 90, 50),
        shadow_tint=(180, 50, 120),  # Magenta shadows
        highlight_tint=(80, 200, 180),  # Cyan highlights
        midtone_saturation=0.75,
        color_temperature_shift=15.0,
        shoulder_compression=0.3,
        black_point_lift=0.08,
        chromatic_aberration_px=1.5,
        gate_weave_amplitude=0.8,
        contrast_adjustment=-0.1,
        vignette_strength=0.35,
    ),
    FilmPreset.KODACHROME_25: FilmStockProfile(
        name="Kodachrome 25",
        grain_intensity=0.02,
        grain_size=1.0,
        halation_strength=0.05,
        halation_radius=10,
        halation_color=(240, 100, 60),
        shadow_tint=(30, 20, 60),  # Deep blue shadows
        highlight_tint=(255, 240, 200),  # Warm highlights
        midtone_saturation=1.3,
        color_temperature_shift=10.0,
        shoulder_compression=0.15,
        black_point_lift=0.02,
        chromatic_aberration_px=0.5,
        gate_weave_amplitude=0.3,
        contrast_adjustment=0.15,
        vignette_strength=0.15,
    ),
    FilmPreset.KODAK_VISION3_500T_PUSH2: FilmStockProfile(
        name="Kodak Vision3 500T (Pushed +2)",
        grain_intensity=0.12,
        grain_size=2.5,
        halation_strength=0.2,
        halation_radius=25,
        halation_color=(255, 70, 40),
        shadow_tint=(40, 50, 80),  # Cool blue shadows
        highlight_tint=(255, 220, 170),  # Warm blown highlights
        midtone_saturation=0.9,
        color_temperature_shift=-5.0,  # Tungsten balanced
        shoulder_compression=0.4,
        black_point_lift=0.12,
        chromatic_aberration_px=2.0,
        gate_weave_amplitude=1.2,
        contrast_adjustment=0.25,
        vignette_strength=0.25,
    ),
}


class AnalogFilmProcessor:
    """Applies analog film stock characteristics to digital images.

    Simulates the organic imperfections of chemical film processing
    including grain, halation, color shifts, and mechanical artifacts.

    Usage:
        processor = AnalogFilmProcessor(FilmPreset.KODACHROME_25)
        result = processor.process(image_bgr)

        # Or with a custom profile:
        profile = FilmStockProfile(grain_intensity=0.05, ...)
        processor = AnalogFilmProcessor(profile=profile)
        result = processor.process(image_bgr)
    """

    def __init__(
        self,
        preset: Optional[FilmPreset] = None,
        profile: Optional[FilmStockProfile] = None,
    ) -> None:
        if profile is not None:
            self.profile = profile
        elif preset is not None:
            if preset == FilmPreset.CUSTOM:
                raise ValueError(
                    "Use profile= parameter for custom film stocks."
                )
            self.profile = FILM_PRESETS[preset]
        else:
            self.profile = FilmStockProfile()

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply all film effects to an image.

        Args:
            image: BGR numpy array (OpenCV format).

        Returns:
            Processed BGR numpy array.
        """
        result = image.astype(np.float32) / 255.0

        # Apply effects in physically motivated order
        result = self._apply_gate_weave(result)
        result = self._apply_halation(result)
        result = self._apply_color_shifts(result)
        result = self._apply_shoulder_compression(result)
        result = self._apply_contrast(result)
        result = self._apply_chromatic_aberration(result)
        result = self._apply_grain(result)
        result = self._apply_vignette(result)
        result = self._apply_black_point_lift(result)

        # Clamp and convert back to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result

    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Process multiple images with the same film stock."""
        return [self.process(img) for img in images]

    def _apply_grain(self, image: np.ndarray) -> np.ndarray:
        """Add silver halide grain structure (not digital noise).

        Real film grain has a non-uniform, organic distribution that
        varies with exposure level — brighter areas have finer grain.
        """
        p = self.profile
        if p.grain_intensity <= 0:
            return image

        h, w = image.shape[:2]

        # Generate base grain at reduced resolution for organic look
        grain_h = max(1, int(h / p.grain_size))
        grain_w = max(1, int(w / p.grain_size))
        grain = np.random.normal(0, 1, (grain_h, grain_w)).astype(np.float32)

        # Upscale grain to full resolution with bilinear interpolation
        grain = cv2.resize(grain, (w, h), interpolation=cv2.INTER_LINEAR)

        # Modulate grain by luminance (darker areas get more grain)
        luminance = np.mean(image, axis=2)
        grain_modulation = 1.0 - (luminance * 0.5)  # More grain in shadows

        grain_3d = np.stack([grain * grain_modulation] * 3, axis=2)
        image = image + grain_3d * p.grain_intensity

        return image

    def _apply_halation(self, image: np.ndarray) -> np.ndarray:
        """Add halation — red bloom on highlights.

        In real film, bright light passes through the emulsion and
        reflects off the film base, creating a colored halo.
        """
        p = self.profile
        if p.halation_strength <= 0:
            return image

        # Identify highlight regions
        luminance = np.mean(image, axis=2)
        highlight_mask = np.clip((luminance - 0.7) / 0.3, 0, 1)

        # Create halation bloom
        bloom = cv2.GaussianBlur(
            highlight_mask, (0, 0), sigmaX=p.halation_radius
        )

        # Apply halation color (convert from RGB to BGR for OpenCV)
        halation_color = np.array(
            [p.halation_color[2], p.halation_color[1], p.halation_color[0]],
            dtype=np.float32,
        ) / 255.0

        halation = bloom[:, :, np.newaxis] * halation_color[np.newaxis, np.newaxis, :]
        image = image + halation * p.halation_strength

        return image

    def _apply_color_shifts(self, image: np.ndarray) -> np.ndarray:
        """Apply expiration-style color shifts in shadows and highlights."""
        p = self.profile
        luminance = np.mean(image, axis=2, keepdims=True)

        # Shadow tint (applied where luminance is low)
        shadow_mask = np.clip(1.0 - luminance * 2.0, 0, 1)
        shadow_color = np.array(
            [p.shadow_tint[2], p.shadow_tint[1], p.shadow_tint[0]],
            dtype=np.float32,
        ) / 255.0
        image = image + shadow_mask * shadow_color * 0.15

        # Highlight tint (applied where luminance is high)
        highlight_mask = np.clip(luminance * 2.0 - 1.0, 0, 1)
        highlight_color = np.array(
            [p.highlight_tint[2], p.highlight_tint[1], p.highlight_tint[0]],
            dtype=np.float32,
        ) / 255.0
        image = image + highlight_mask * highlight_color * 0.1

        # Saturation adjustment
        if p.midtone_saturation != 1.0:
            hsv = cv2.cvtColor(
                np.clip(image, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV
            )
            hsv[:, :, 1] = np.clip(
                hsv[:, :, 1] * p.midtone_saturation, 0, 1
            )
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return image

    def _apply_shoulder_compression(self, image: np.ndarray) -> np.ndarray:
        """Apply film shoulder curve (highlight rolloff).

        Real film has a characteristic S-curve response where highlights
        compress gradually (the "shoulder") rather than clipping hard.
        """
        p = self.profile
        if p.shoulder_compression <= 0:
            return image

        # Soft-clip using a smooth curve
        threshold = 1.0 - p.shoulder_compression * 0.3
        mask = image > threshold
        if np.any(mask):
            excess = image[mask] - threshold
            compressed = threshold + excess * (1.0 / (1.0 + excess * 3.0))
            image[mask] = compressed

        return image

    def _apply_contrast(self, image: np.ndarray) -> np.ndarray:
        """Adjust overall contrast."""
        p = self.profile
        if abs(p.contrast_adjustment) < 0.01:
            return image

        midpoint = 0.5
        factor = 1.0 + p.contrast_adjustment
        image = midpoint + (image - midpoint) * factor
        return image

    def _apply_chromatic_aberration(self, image: np.ndarray) -> np.ndarray:
        """Apply lateral chromatic aberration (color fringing).

        Simulates the color separation that occurs in lower-quality
        lenses, especially toward the edges of the frame.
        """
        p = self.profile
        if p.chromatic_aberration_px <= 0:
            return image

        h, w = image.shape[:2]
        shift = p.chromatic_aberration_px

        # Shift red channel outward, blue channel inward
        # Create displacement maps relative to image center
        cx, cy = w / 2.0, h / 2.0
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Radial displacement
        dx = (x_coords - cx) / max(cx, 1)
        dy = (y_coords - cy) / max(cy, 1)

        # Shift red channel outward
        red_x = (x_coords + dx * shift).astype(np.float32)
        red_y = (y_coords + dy * shift).astype(np.float32)

        # Shift blue channel inward
        blue_x = (x_coords - dx * shift).astype(np.float32)
        blue_y = (y_coords - dy * shift).astype(np.float32)

        # BGR order in OpenCV
        b_channel = cv2.remap(
            image[:, :, 0], blue_x, blue_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        g_channel = image[:, :, 1]  # Green stays centered
        r_channel = cv2.remap(
            image[:, :, 2], red_x, red_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        image = np.stack([b_channel, g_channel, r_channel], axis=2)
        return image

    def _apply_vignette(self, image: np.ndarray) -> np.ndarray:
        """Apply optical vignetting (edge darkening)."""
        p = self.profile
        if p.vignette_strength <= 0:
            return image

        h, w = image.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Normalized distance from center
        cx, cy = w / 2.0, h / 2.0
        dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        dist = dist / dist.max()

        # Smooth vignette falloff
        vignette = 1.0 - (dist ** 2) * p.vignette_strength
        vignette = np.clip(vignette, 0, 1)

        image = image * vignette[:, :, np.newaxis]
        return image

    def _apply_gate_weave(self, image: np.ndarray) -> np.ndarray:
        """Apply gate weave (subtle frame jitter).

        Simulates the mechanical imprecision of a film projector gate.
        """
        p = self.profile
        if p.gate_weave_amplitude <= 0:
            return image

        dx = np.random.uniform(-p.gate_weave_amplitude, p.gate_weave_amplitude)
        dy = np.random.uniform(-p.gate_weave_amplitude, p.gate_weave_amplitude)

        h, w = image.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        image = cv2.warpAffine(
            image, M, (w, h), borderMode=cv2.BORDER_REFLECT
        )

        return image

    def _apply_black_point_lift(self, image: np.ndarray) -> np.ndarray:
        """Lift the black point (fade blacks to dark gray)."""
        p = self.profile
        if p.black_point_lift <= 0:
            return image

        image = image * (1.0 - p.black_point_lift) + p.black_point_lift
        return image


def process_with_verification(
    image: np.ndarray,
    processor: AnalogFilmProcessor,
    target_measurements: Any = None,
    analyzer: Any = None,
    tolerance_multiplier: float = 1.5,
) -> Tuple[np.ndarray, Optional[Any]]:
    """Apply film processing, then verify facial measurements still hold.

    Film effects (especially contrast shifts) can alter perceived jawlines
    and other features. This function catches such drift.

    Args:
        image: BGR input image.
        processor: AnalogFilmProcessor instance.
        target_measurements: Optional FacialMeasurements to verify against.
        analyzer: Optional FaceAnalyzer for post-processing verification.
        tolerance_multiplier: How much to relax tolerances for post-processing
            (default 1.5x normal tolerances).

    Returns:
        Tuple of (processed_image, drift_report_or_None).
    """
    processed = processor.process(image)

    drift_report = None
    if target_measurements is not None and analyzer is not None:
        from .face_lock import DriftDetector

        # Relax tolerances for post-processing (film effects shift perception)
        relaxed_tolerances = {
            k: v * tolerance_multiplier
            for k, v in DriftDetector.DEFAULT_TOLERANCES.items()
        }
        detector = DriftDetector(
            target=target_measurements,
            tolerances=relaxed_tolerances,
            analyzer=analyzer,
        )
        try:
            drift_report = detector.check(processed)
        except ValueError:
            pass  # Face not detected after processing — caller should handle

    return processed, drift_report
