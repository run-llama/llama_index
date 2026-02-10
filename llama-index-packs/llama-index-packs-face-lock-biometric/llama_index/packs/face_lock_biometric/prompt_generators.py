"""
Prompt generators for AI image generation platforms.

Converts FacialMeasurements into structured prompts optimized for
Reve, Flux, and Midjourney. Includes both positive prompts (what to
generate) and negative prompts (drift prevention).
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

from .face_lock import EyeShape, FacialMeasurements, FitzpatrickScale


class Platform(Enum):
    """Supported AI image generation platforms."""

    REVE = "reve"
    FLUX = "flux"
    MIDJOURNEY = "midjourney"
    GENERIC = "generic"


# Mapping from Fitzpatrick scale to descriptive skin tone terms
_SKIN_TONE_MAP: Dict[FitzpatrickScale, str] = {
    FitzpatrickScale.TYPE_I: "very fair porcelain skin",
    FitzpatrickScale.TYPE_II: "fair skin with light undertones",
    FitzpatrickScale.TYPE_III: "medium skin with warm undertones",
    FitzpatrickScale.TYPE_IV: "olive skin with golden undertones",
    FitzpatrickScale.TYPE_V: "brown skin with rich warm undertones",
    FitzpatrickScale.TYPE_VI: "deep dark brown skin with cool undertones",
}

# Mapping from eye shape classification to descriptive terms
_EYE_SHAPE_MAP: Dict[EyeShape, str] = {
    EyeShape.ALMOND: "almond-shaped eyes",
    EyeShape.ROUND: "round wide-set eyes",
    EyeShape.HOODED: "hooded eyes with heavy lids",
    EyeShape.MONOLID: "monolid eyes",
    EyeShape.DOWNTURNED: "downturned eyes",
    EyeShape.UPTURNED: "upturned cat eyes",
}


def _jawline_descriptor(gonial_angle: float) -> str:
    """Convert gonial angle to descriptive jawline term."""
    if gonial_angle < 120:
        return "extremely sharp chiseled jawline"
    elif gonial_angle < 125:
        return "sharp defined jawline"
    elif gonial_angle < 130:
        return "moderately defined jawline"
    elif gonial_angle < 135:
        return "soft rounded jawline"
    else:
        return "wide rounded jawline"


def _face_shape_descriptor(facial_index: float) -> str:
    """Convert facial index to face shape descriptor."""
    if facial_index < 1.2:
        return "wide round face shape"
    elif facial_index < 1.3:
        return "oval face shape"
    elif facial_index < 1.4:
        return "oblong face shape"
    else:
        return "long narrow face shape"


def _nose_descriptor(nasal_rotation: float) -> str:
    """Convert nasal tip rotation to nose descriptor."""
    if nasal_rotation < 90:
        return "slightly drooping nose tip"
    elif nasal_rotation < 95:
        return "straight nose profile"
    elif nasal_rotation < 100:
        return "slightly upturned nose"
    elif nasal_rotation < 105:
        return "upturned button nose"
    else:
        return "strongly upturned nose"


def _cheekbone_descriptor(zygomatic: float) -> str:
    """Convert zygomatic prominence to cheekbone descriptor."""
    if zygomatic < 0.12:
        return "flat subtle cheekbones"
    elif zygomatic < 0.16:
        return "moderately defined cheekbones"
    elif zygomatic < 0.20:
        return "prominent high cheekbones"
    else:
        return "very prominent sculpted cheekbones"


def _canthal_descriptor(canthal_tilt: float) -> str:
    """Convert canthal tilt to eye tilt descriptor."""
    if canthal_tilt < -3:
        return "downward-tilted eye corners"
    elif canthal_tilt < 0:
        return "slightly downturned eye corners"
    elif canthal_tilt < 3:
        return "neutral horizontal eye axis"
    elif canthal_tilt < 5:
        return "slightly upturned eye corners"
    else:
        return "strongly upturned fox eye shape"


def _lip_descriptor(lip_ratio: float) -> str:
    """Convert lip fullness ratio to lip descriptor."""
    if lip_ratio < 0.4:
        return "thin upper lip with fuller lower lip"
    elif lip_ratio < 0.6:
        return "balanced medium lips"
    elif lip_ratio < 0.8:
        return "full lips"
    else:
        return "very full prominent lips"


class PromptGenerator:
    """Generates AI image generation prompts from facial measurements.

    Converts numeric biometric data into natural language descriptors
    optimized for specific platforms.

    Usage:
        gen = PromptGenerator(platform=Platform.REVE)
        positive, negative = gen.generate(measurements)
        print(f"Positive prompt: {positive}")
        print(f"Negative prompt: {negative}")
    """

    def __init__(
        self,
        platform: Platform = Platform.GENERIC,
        style_prefix: str = "",
        style_suffix: str = "",
        include_technical: bool = False,
    ) -> None:
        """
        Args:
            platform: Target AI platform for prompt optimization.
            style_prefix: Text prepended to every prompt.
            style_suffix: Text appended to every prompt.
            include_technical: If True, include raw numeric values as
                comments in the prompt (useful for debugging).
        """
        self.platform = platform
        self.style_prefix = style_prefix
        self.style_suffix = style_suffix
        self.include_technical = include_technical

    def generate(
        self,
        measurements: FacialMeasurements,
        additional_context: str = "",
    ) -> Tuple[str, str]:
        """Generate positive and negative prompts from measurements.

        Args:
            measurements: FacialMeasurements to convert.
            additional_context: Extra context to append (scene, clothing, etc.).

        Returns:
            Tuple of (positive_prompt, negative_prompt).
        """
        positive = self._build_positive(measurements, additional_context)
        negative = self._build_negative(measurements)
        return positive, negative

    def generate_character_prompt(
        self,
        measurements: FacialMeasurements,
        character_name: str = "",
        scene_description: str = "",
        camera_angle: str = "front-facing portrait",
        lighting: str = "natural lighting",
    ) -> Tuple[str, str]:
        """Generate a complete character prompt with scene context.

        Args:
            measurements: Target facial measurements.
            character_name: Optional character name for reference.
            scene_description: Scene/background description.
            camera_angle: Camera angle descriptor.
            lighting: Lighting descriptor.

        Returns:
            Tuple of (positive_prompt, negative_prompt).
        """
        context_parts = [camera_angle, lighting]
        if scene_description:
            context_parts.append(scene_description)
        context = ", ".join(context_parts)

        positive, negative = self.generate(measurements, context)

        if character_name and self.platform == Platform.MIDJOURNEY:
            # Midjourney benefits from character labels
            positive = f'"{character_name}" {positive}'

        return positive, negative

    def _build_positive(
        self, m: FacialMeasurements, context: str = ""
    ) -> str:
        """Build the positive prompt."""
        parts: List[str] = []

        if self.style_prefix:
            parts.append(self.style_prefix)

        # Core facial descriptors
        parts.append(_face_shape_descriptor(m.facial_index))
        parts.append(_jawline_descriptor(m.gonial_angle))
        parts.append(_cheekbone_descriptor(m.zygomatic_prominence))
        parts.append(_EYE_SHAPE_MAP.get(m.eye_shape, "almond-shaped eyes"))
        parts.append(_canthal_descriptor(m.canthal_tilt))
        parts.append(_nose_descriptor(m.nasal_tip_rotation))
        parts.append(_lip_descriptor(m.lip_fullness_ratio))
        parts.append(
            _SKIN_TONE_MAP.get(m.fitzpatrick_type, "medium skin tone")
        )

        # Symmetry note (only mention if notably symmetric or asymmetric)
        if m.face_symmetry_score > 0.95:
            parts.append("highly symmetrical facial features")
        elif m.face_symmetry_score < 0.85:
            parts.append("slightly asymmetrical natural features")

        if context:
            parts.append(context)

        if self.style_suffix:
            parts.append(self.style_suffix)

        # Platform-specific formatting
        if self.platform == Platform.REVE:
            prompt = ", ".join(parts)
            prompt = f"photorealistic portrait, {prompt}, 8k, detailed skin texture"
        elif self.platform == Platform.FLUX:
            prompt = ", ".join(parts)
            prompt = f"photo, {prompt}, high detail, sharp focus"
        elif self.platform == Platform.MIDJOURNEY:
            prompt = ", ".join(parts)
            prompt = f"{prompt} --v 6 --style raw"
        else:
            prompt = ", ".join(parts)

        if self.include_technical:
            tech = (
                f" [BIOMETRICS: jaw={m.gonial_angle:.1f}°, "
                f"index={m.facial_index:.3f}, "
                f"canthal={m.canthal_tilt:.1f}°, "
                f"nasal={m.nasal_tip_rotation:.1f}°, "
                f"zygo={m.zygomatic_prominence:.3f}]"
            )
            prompt += tech

        return prompt

    def _build_negative(self, m: FacialMeasurements) -> str:
        """Build the negative prompt for drift prevention."""
        negatives: List[str] = []

        # Always include standard quality negatives
        negatives.extend([
            "deformed face",
            "asymmetric eyes",
            "blurry",
            "low quality",
            "distorted features",
        ])

        # Specific anti-drift negatives based on measurements
        if m.gonial_angle < 125:
            # Sharp jaw — prevent rounding
            negatives.append("round jaw")
            negatives.append("soft jawline")
            negatives.append("double chin")
        elif m.gonial_angle > 133:
            # Round jaw — prevent sharpening
            negatives.append("sharp jaw")
            negatives.append("angular jawline")
            negatives.append("chiseled jaw")

        if m.facial_index > 1.35:
            # Long face — prevent widening
            negatives.append("wide face")
            negatives.append("round face")
        elif m.facial_index < 1.25:
            # Wide face — prevent lengthening
            negatives.append("long face")
            negatives.append("narrow face")

        if m.canthal_tilt > 3:
            negatives.append("droopy eyes")
            negatives.append("downturned eyes")
        elif m.canthal_tilt < -3:
            negatives.append("cat eyes")
            negatives.append("upturned eyes")

        if m.nasal_tip_rotation < 95:
            negatives.append("upturned nose")
            negatives.append("button nose")
        elif m.nasal_tip_rotation > 100:
            negatives.append("drooping nose")
            negatives.append("hooked nose")

        # Platform-specific negative formatting
        if self.platform == Platform.MIDJOURNEY:
            return " --no " + ", ".join(negatives)
        else:
            return ", ".join(negatives)

    def format_for_api(
        self, measurements: FacialMeasurements, **kwargs: str
    ) -> Dict[str, str]:
        """Return a dict suitable for API submission.

        Returns:
            Dict with 'positive_prompt', 'negative_prompt', and 'platform'.
        """
        positive, negative = self.generate(measurements, **kwargs)
        return {
            "positive_prompt": positive,
            "negative_prompt": negative,
            "platform": self.platform.value,
        }


def generate_prompt_for_platform(
    measurements: FacialMeasurements,
    platform: Platform = Platform.GENERIC,
    context: str = "",
    include_technical: bool = False,
) -> Tuple[str, str]:
    """Convenience function to generate prompts without instantiating a class.

    Args:
        measurements: Target facial measurements.
        platform: Target AI platform.
        context: Additional context (scene, etc.).
        include_technical: Include raw biometric values in prompt.

    Returns:
        Tuple of (positive_prompt, negative_prompt).
    """
    gen = PromptGenerator(
        platform=platform, include_technical=include_technical
    )
    return gen.generate(measurements, additional_context=context)
