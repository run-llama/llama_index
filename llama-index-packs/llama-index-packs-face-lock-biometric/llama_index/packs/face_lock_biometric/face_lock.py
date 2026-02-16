"""
Face Lock: Core facial measurement engine.

Extracts precise biometric measurements from reference images using MediaPipe's
468-landmark face mesh. Measurements include gonial angle, canthal tilt, facial
index, nasal tip rotation, zygomatic prominence, and eye shape classification.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError(
        "opencv-python is required for Face Lock. "
        "Install with: pip install opencv-python"
    )

try:
    import mediapipe as mp
except ImportError:
    raise ImportError(
        "mediapipe is required for Face Lock. "
        "Install with: pip install mediapipe"
    )


class EyeShape(Enum):
    """Eye shape classification."""

    ALMOND = "almond"
    ROUND = "round"
    HOODED = "hooded"
    MONOLID = "monolid"
    DOWNTURNED = "downturned"
    UPTURNED = "upturned"


class FitzpatrickScale(Enum):
    """Fitzpatrick skin type classification (I-VI)."""

    TYPE_I = 1  # Very fair, always burns
    TYPE_II = 2  # Fair, usually burns
    TYPE_III = 3  # Medium, sometimes burns
    TYPE_IV = 4  # Olive, rarely burns
    TYPE_V = 5  # Brown, very rarely burns
    TYPE_VI = 6  # Dark brown/black, never burns


@dataclass
class FacialMeasurements:
    """
    Structured biometric measurements extracted from a face image.

    All angles are in degrees. Distances are normalized to inter-pupillary
    distance (IPD) to be scale-invariant.
    """

    # Core geometric measurements
    gonial_angle: float = 0.0  # Jawline sharpness: 120-135° typical
    facial_index: float = 0.0  # Face length/width ratio: 1.2-1.4 typical
    canthal_tilt: float = 0.0  # Eye angle: ±5° typical
    nasal_tip_rotation: float = 0.0  # Nose tip angle: 90-105° typical
    zygomatic_prominence: float = 0.0  # Cheekbone projection (normalized)

    # Classification attributes
    eye_shape: EyeShape = EyeShape.ALMOND
    fitzpatrick_type: FitzpatrickScale = FitzpatrickScale.TYPE_III

    # Supplementary measurements
    philtrum_length: float = 0.0  # Upper lip to nose base (normalized)
    lip_fullness_ratio: float = 0.0  # Upper/lower lip height ratio
    brow_arch_height: float = 0.0  # Eyebrow peak height (normalized)
    face_symmetry_score: float = 0.0  # 0-1, where 1 is perfect symmetry
    interpupillary_distance_px: float = 0.0  # IPD in pixels (for reference)

    # Raw landmark data (optional, for advanced use)
    raw_landmarks: Optional[List[Tuple[float, float, float]]] = field(
        default=None, repr=False
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert measurements to a dictionary (excluding raw landmarks)."""
        return {
            "gonial_angle": round(self.gonial_angle, 1),
            "facial_index": round(self.facial_index, 3),
            "canthal_tilt": round(self.canthal_tilt, 1),
            "nasal_tip_rotation": round(self.nasal_tip_rotation, 1),
            "zygomatic_prominence": round(self.zygomatic_prominence, 3),
            "eye_shape": self.eye_shape.value,
            "fitzpatrick_type": self.fitzpatrick_type.value,
            "philtrum_length": round(self.philtrum_length, 3),
            "lip_fullness_ratio": round(self.lip_fullness_ratio, 3),
            "brow_arch_height": round(self.brow_arch_height, 3),
            "face_symmetry_score": round(self.face_symmetry_score, 3),
            "interpupillary_distance_px": round(
                self.interpupillary_distance_px, 1
            ),
        }


# MediaPipe face mesh landmark indices for key facial points.
# See: https://github.com/google/mediapipe/blob/master/mediapipe/modules/
#      face_geometry/data/canonical_face_model_uv_visualization.png
class _LandmarkIndices:
    """MediaPipe 468-landmark face mesh indices for key facial features."""

    # Eyes
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    # Pupils (iris center approximations)
    LEFT_PUPIL = 468  # Iris landmarks (468-472 for left)
    RIGHT_PUPIL = 473  # Iris landmarks (473-477 for right)
    # Fallback pupil approximations using eye corners
    LEFT_PUPIL_APPROX_INNER = 133
    LEFT_PUPIL_APPROX_OUTER = 33
    RIGHT_PUPIL_APPROX_INNER = 362
    RIGHT_PUPIL_APPROX_OUTER = 263

    # Jaw
    JAW_LEFT = 234  # Left jaw angle (gonion)
    JAW_RIGHT = 454  # Right jaw angle (gonion)
    JAW_TIP = 152  # Chin / menton
    JAW_LEFT_RAMUS_TOP = 132  # Top of left jaw ramus
    JAW_RIGHT_RAMUS_TOP = 361  # Top of right jaw ramus

    # Nose
    NOSE_TIP = 1
    NOSE_BRIDGE_TOP = 6  # Nasion
    NOSE_BASE_LEFT = 129  # Left alar base
    NOSE_BASE_RIGHT = 358  # Right alar base
    NOSE_COLUMELLA = 2  # Base of nose between nostrils

    # Face outline
    FOREHEAD_TOP = 10  # Top of face
    CHIN_BOTTOM = 152  # Bottom of face

    # Zygomatic (cheekbones)
    LEFT_ZYGOMATIC = 123
    RIGHT_ZYGOMATIC = 352
    LEFT_CHEEK_OUTER = 116
    RIGHT_CHEEK_OUTER = 345

    # Lips
    UPPER_LIP_TOP = 13
    UPPER_LIP_BOTTOM = 14
    LOWER_LIP_TOP = 14
    LOWER_LIP_BOTTOM = 17

    # Philtrum
    PHILTRUM_TOP = 2  # Subnasale (nose base)
    PHILTRUM_BOTTOM = 13  # Upper lip vermilion border

    # Eyebrows
    LEFT_BROW_INNER = 107
    LEFT_BROW_PEAK = 105
    LEFT_BROW_OUTER = 70
    RIGHT_BROW_INNER = 336
    RIGHT_BROW_PEAK = 334
    RIGHT_BROW_OUTER = 300

    # Face width reference points
    FACE_WIDTH_LEFT = 234
    FACE_WIDTH_RIGHT = 454


def _angle_between_points(
    p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray
) -> float:
    """Calculate the angle at vertex formed by rays to p1 and p2 (degrees)."""
    v1 = p1 - vertex
    v2 = p2 - vertex
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def _midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Midpoint between two points."""
    return (p1 + p2) / 2.0


class FaceAnalyzer:
    """
    Extracts precise biometric measurements from face images.

    Uses MediaPipe's 468-landmark face mesh for sub-pixel accuracy.

    Usage:
        analyzer = FaceAnalyzer()
        measurements = analyzer.analyze("reference.jpg")
        print(f"Gonial angle: {measurements.gonial_angle}°")
        print(f"Facial index: {measurements.facial_index}")
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def analyze(self, image_input: Any) -> FacialMeasurements:
        """
        Analyze a face image and return biometric measurements.

        Args:
            image_input: File path (str), numpy array (BGR), or PIL Image.

        Returns:
            FacialMeasurements dataclass with all extracted measurements.

        Raises:
            ValueError: If no face is detected in the image.

        """
        image_rgb, image_bgr = self._load_image(image_input)
        h, w = image_rgb.shape[:2]

        results = self._face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            raise ValueError(
                "No face detected in the image. Ensure the image contains "
                "a clearly visible frontal face."
            )

        face_landmarks = results.multi_face_landmarks[0]
        landmarks_3d = self._extract_landmark_array(face_landmarks, w, h)

        measurements = FacialMeasurements()
        measurements.raw_landmarks = [
            (lm.x, lm.y, lm.z) for lm in face_landmarks.landmark
        ]

        # Calculate inter-pupillary distance for normalization
        ipd = self._calculate_ipd(landmarks_3d)
        measurements.interpupillary_distance_px = ipd

        # Core measurements
        measurements.gonial_angle = self._calculate_gonial_angle(landmarks_3d)
        measurements.facial_index = self._calculate_facial_index(landmarks_3d)
        measurements.canthal_tilt = self._calculate_canthal_tilt(landmarks_3d)
        measurements.nasal_tip_rotation = self._calculate_nasal_tip_rotation(
            landmarks_3d
        )
        measurements.zygomatic_prominence = (
            self._calculate_zygomatic_prominence(landmarks_3d, ipd)
        )

        # Classifications
        measurements.eye_shape = self._classify_eye_shape(landmarks_3d)
        measurements.fitzpatrick_type = self._estimate_fitzpatrick(image_bgr, landmarks_3d)

        # Supplementary measurements
        measurements.philtrum_length = self._calculate_philtrum_length(
            landmarks_3d, ipd
        )
        measurements.lip_fullness_ratio = self._calculate_lip_fullness(
            landmarks_3d
        )
        measurements.brow_arch_height = self._calculate_brow_arch(
            landmarks_3d, ipd
        )
        measurements.face_symmetry_score = self._calculate_symmetry(
            landmarks_3d
        )

        return measurements

    def analyze_multiple(
        self, image_inputs: List[Any]
    ) -> List[FacialMeasurements]:
        """Analyze multiple images and return measurements for each."""
        return [self.analyze(img) for img in image_inputs]

    def _load_image(self, image_input: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image from file path, numpy array, or PIL Image.

        Returns (RGB array, BGR array).
        """
        if isinstance(image_input, str):
            image_bgr = cv2.imread(image_input)
            if image_bgr is None:
                raise ValueError(f"Could not load image from path: {image_input}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image_bgr = image_input
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Image array must be 3-channel (BGR).")
        else:
            # Assume PIL Image
            try:
                image_rgb = np.array(image_input)
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                raise ValueError(
                    f"Unsupported image input type: {type(image_input)}"
                )

        return image_rgb, image_bgr

    def _extract_landmark_array(
        self, face_landmarks: Any, w: int, h: int
    ) -> np.ndarray:
        """Convert MediaPipe landmarks to numpy array of pixel coordinates."""
        landmarks = []
        for lm in face_landmarks.landmark:
            landmarks.append([lm.x * w, lm.y * h, lm.z * w])
        return np.array(landmarks)

    def _calculate_ipd(self, landmarks: np.ndarray) -> float:
        """Calculate inter-pupillary distance."""
        LI = _LandmarkIndices
        # Use iris landmarks if available (indices 468+), otherwise approximate
        if len(landmarks) > LI.RIGHT_PUPIL:
            left_pupil = landmarks[LI.LEFT_PUPIL][:2]
            right_pupil = landmarks[LI.RIGHT_PUPIL][:2]
        else:
            left_pupil = _midpoint(
                landmarks[LI.LEFT_PUPIL_APPROX_INNER][:2],
                landmarks[LI.LEFT_PUPIL_APPROX_OUTER][:2],
            )
            right_pupil = _midpoint(
                landmarks[LI.RIGHT_PUPIL_APPROX_INNER][:2],
                landmarks[LI.RIGHT_PUPIL_APPROX_OUTER][:2],
            )
        return _distance(left_pupil, right_pupil)

    def _calculate_gonial_angle(self, landmarks: np.ndarray) -> float:
        """
        Calculate the gonial (jaw) angle.

        Measured at the jaw angle (gonion) between the ramus and body of
        the mandible. Normal range: 120-135°.
        """
        LI = _LandmarkIndices
        # Average left and right gonial angles
        left_angle = _angle_between_points(
            landmarks[LI.JAW_LEFT_RAMUS_TOP][:2],
            landmarks[LI.JAW_LEFT][:2],
            landmarks[LI.JAW_TIP][:2],
        )
        right_angle = _angle_between_points(
            landmarks[LI.JAW_RIGHT_RAMUS_TOP][:2],
            landmarks[LI.JAW_RIGHT][:2],
            landmarks[LI.JAW_TIP][:2],
        )
        return (left_angle + right_angle) / 2.0

    def _calculate_facial_index(self, landmarks: np.ndarray) -> float:
        """
        Calculate facial index (face height / face width ratio).

        Normal range: 1.2-1.4 (leptoprosopic to mesoprosopic).
        """
        LI = _LandmarkIndices
        face_height = _distance(
            landmarks[LI.FOREHEAD_TOP][:2],
            landmarks[LI.CHIN_BOTTOM][:2],
        )
        face_width = _distance(
            landmarks[LI.FACE_WIDTH_LEFT][:2],
            landmarks[LI.FACE_WIDTH_RIGHT][:2],
        )
        if face_width < 1e-8:
            return 0.0
        return face_height / face_width

    def _calculate_canthal_tilt(self, landmarks: np.ndarray) -> float:
        """
        Calculate canthal tilt (eye angle).

        Positive = upward tilt (lateral canthus higher than medial).
        Normal range: ±5°.
        """
        LI = _LandmarkIndices
        # Left eye
        left_dx = landmarks[LI.LEFT_EYE_OUTER][0] - landmarks[LI.LEFT_EYE_INNER][0]
        left_dy = landmarks[LI.LEFT_EYE_OUTER][1] - landmarks[LI.LEFT_EYE_INNER][1]
        left_tilt = math.degrees(math.atan2(-left_dy, abs(left_dx)))

        # Right eye
        right_dx = (
            landmarks[LI.RIGHT_EYE_INNER][0] - landmarks[LI.RIGHT_EYE_OUTER][0]
        )
        right_dy = (
            landmarks[LI.RIGHT_EYE_INNER][1] - landmarks[LI.RIGHT_EYE_OUTER][1]
        )
        right_tilt = math.degrees(math.atan2(-right_dy, abs(right_dx)))

        return (left_tilt + right_tilt) / 2.0

    def _calculate_nasal_tip_rotation(self, landmarks: np.ndarray) -> float:
        """
        Calculate nasolabial angle (nasal tip rotation).

        Angle between columella and upper lip. Normal range: 90-105°.
        """
        LI = _LandmarkIndices
        return _angle_between_points(
            landmarks[LI.NOSE_BRIDGE_TOP][:2],
            landmarks[LI.NOSE_TIP][:2],
            landmarks[LI.NOSE_COLUMELLA][:2],
        )

    def _calculate_zygomatic_prominence(
        self, landmarks: np.ndarray, ipd: float
    ) -> float:
        """
        Calculate zygomatic (cheekbone) prominence.

        Measured as the lateral projection of the cheekbone normalized
        by inter-pupillary distance.
        """
        LI = _LandmarkIndices
        if ipd < 1e-8:
            return 0.0

        left_projection = _distance(
            landmarks[LI.LEFT_ZYGOMATIC][:2],
            landmarks[LI.LEFT_CHEEK_OUTER][:2],
        )
        right_projection = _distance(
            landmarks[LI.RIGHT_ZYGOMATIC][:2],
            landmarks[LI.RIGHT_CHEEK_OUTER][:2],
        )
        avg_projection = (left_projection + right_projection) / 2.0
        return avg_projection / ipd

    def _classify_eye_shape(self, landmarks: np.ndarray) -> EyeShape:
        """Classify eye shape based on geometric relationships."""
        LI = _LandmarkIndices

        # Calculate eye aspect ratio (height/width)
        left_width = _distance(
            landmarks[LI.LEFT_EYE_INNER][:2],
            landmarks[LI.LEFT_EYE_OUTER][:2],
        )
        left_height = _distance(
            landmarks[LI.LEFT_EYE_TOP][:2],
            landmarks[LI.LEFT_EYE_BOTTOM][:2],
        )

        if left_width < 1e-8:
            return EyeShape.ALMOND

        aspect_ratio = left_height / left_width
        canthal = self._calculate_canthal_tilt(landmarks)

        # Classification heuristics
        if aspect_ratio > 0.38:
            return EyeShape.ROUND
        elif aspect_ratio < 0.22:
            return EyeShape.MONOLID
        elif canthal > 4.0:
            return EyeShape.UPTURNED
        elif canthal < -4.0:
            return EyeShape.DOWNTURNED
        else:
            # Check for hooded by comparing brow-to-lid distance
            brow_height = landmarks[LI.LEFT_BROW_PEAK][1]
            lid_height = landmarks[LI.LEFT_EYE_TOP][1]
            brow_lid_dist = abs(lid_height - brow_height)
            if brow_lid_dist < left_height * 0.5:
                return EyeShape.HOODED
            return EyeShape.ALMOND

    def _estimate_fitzpatrick(
        self, image_bgr: np.ndarray, landmarks: np.ndarray
    ) -> FitzpatrickScale:
        """
        Estimate Fitzpatrick skin type from facial skin tone.

        Uses the cheek and forehead regions to sample skin color, then
        maps to Fitzpatrick scale using ITA (Individual Typology Angle).
        """
        LI = _LandmarkIndices
        h, w = image_bgr.shape[:2]

        # Sample skin from cheek regions
        sample_points = [
            landmarks[LI.LEFT_ZYGOMATIC][:2].astype(int),
            landmarks[LI.RIGHT_ZYGOMATIC][:2].astype(int),
        ]

        lab_values = []
        for pt in sample_points:
            x, y = int(np.clip(pt[0], 5, w - 6)), int(np.clip(pt[1], 5, h - 6))
            patch = image_bgr[y - 5 : y + 5, x - 5 : x + 5]
            if patch.size == 0:
                continue
            lab_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
            lab_values.append(lab_patch.mean(axis=(0, 1)))

        if not lab_values:
            return FitzpatrickScale.TYPE_III

        avg_lab = np.mean(lab_values, axis=0)
        L, _a, b = avg_lab[0], avg_lab[1], avg_lab[2]

        # ITA (Individual Typology Angle) formula
        # ITA = arctan((L - 50) / b) * 180 / pi
        if abs(b) < 1e-8:
            ita = 90.0 if L > 50 else -90.0
        else:
            ita = math.degrees(math.atan2(L - 50, b))

        # Map ITA to Fitzpatrick scale
        if ita > 55:
            return FitzpatrickScale.TYPE_I
        elif ita > 41:
            return FitzpatrickScale.TYPE_II
        elif ita > 28:
            return FitzpatrickScale.TYPE_III
        elif ita > 10:
            return FitzpatrickScale.TYPE_IV
        elif ita > -30:
            return FitzpatrickScale.TYPE_V
        else:
            return FitzpatrickScale.TYPE_VI

    def _calculate_philtrum_length(
        self, landmarks: np.ndarray, ipd: float
    ) -> float:
        """Calculate philtrum length normalized by IPD."""
        LI = _LandmarkIndices
        if ipd < 1e-8:
            return 0.0
        length = _distance(
            landmarks[LI.PHILTRUM_TOP][:2],
            landmarks[LI.PHILTRUM_BOTTOM][:2],
        )
        return length / ipd

    def _calculate_lip_fullness(self, landmarks: np.ndarray) -> float:
        """Calculate lip fullness ratio (upper lip height / lower lip height)."""
        LI = _LandmarkIndices
        upper_height = _distance(
            landmarks[LI.UPPER_LIP_TOP][:2],
            landmarks[LI.UPPER_LIP_BOTTOM][:2],
        )
        lower_height = _distance(
            landmarks[LI.LOWER_LIP_TOP][:2],
            landmarks[LI.LOWER_LIP_BOTTOM][:2],
        )
        if lower_height < 1e-8:
            return 0.0
        return upper_height / lower_height

    def _calculate_brow_arch(
        self, landmarks: np.ndarray, ipd: float
    ) -> float:
        """Calculate eyebrow arch height normalized by IPD."""
        LI = _LandmarkIndices
        if ipd < 1e-8:
            return 0.0

        # Height of brow peak above the line connecting inner and outer brow
        left_baseline = _midpoint(
            landmarks[LI.LEFT_BROW_INNER][:2],
            landmarks[LI.LEFT_BROW_OUTER][:2],
        )
        left_arch = abs(landmarks[LI.LEFT_BROW_PEAK][1] - left_baseline[1])

        right_baseline = _midpoint(
            landmarks[LI.RIGHT_BROW_INNER][:2],
            landmarks[LI.RIGHT_BROW_OUTER][:2],
        )
        right_arch = abs(landmarks[LI.RIGHT_BROW_PEAK][1] - right_baseline[1])

        avg_arch = (left_arch + right_arch) / 2.0
        return avg_arch / ipd

    def _calculate_symmetry(self, landmarks: np.ndarray) -> float:
        """
        Calculate facial symmetry score (0-1, 1 = perfectly symmetric).

        Compares left and right side landmark positions relative to the
        vertical midline.
        """
        LI = _LandmarkIndices
        # Define symmetric pairs
        pairs = [
            (LI.LEFT_EYE_INNER, LI.RIGHT_EYE_INNER),
            (LI.LEFT_EYE_OUTER, LI.RIGHT_EYE_OUTER),
            (LI.JAW_LEFT, LI.JAW_RIGHT),
            (LI.LEFT_ZYGOMATIC, LI.RIGHT_ZYGOMATIC),
            (LI.LEFT_BROW_PEAK, LI.RIGHT_BROW_PEAK),
            (LI.NOSE_BASE_LEFT, LI.NOSE_BASE_RIGHT),
        ]

        # Midline reference
        midline_x = landmarks[LI.NOSE_TIP][0]

        deviations = []
        for left_idx, right_idx in pairs:
            left_dist = abs(landmarks[left_idx][0] - midline_x)
            right_dist = abs(landmarks[right_idx][0] - midline_x)
            max_dist = max(left_dist, right_dist)
            if max_dist < 1e-8:
                deviations.append(1.0)
            else:
                symmetry = 1.0 - abs(left_dist - right_dist) / max_dist
                deviations.append(max(0.0, symmetry))

        return float(np.mean(deviations))

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._face_mesh.close()

    def __enter__(self) -> "FaceAnalyzer":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@dataclass
class DriftReport:
    """Report on how much a generated image has drifted from target measurements."""

    image_index: int
    deviations: Dict[str, float]  # measurement_name -> absolute deviation
    overall_drift_score: float  # 0-1 normalized drift
    is_within_tolerance: bool
    details: List[str]  # Human-readable deviation descriptions

    def __str__(self) -> str:
        status = "PASS" if self.is_within_tolerance else "DRIFT DETECTED"
        lines = [f"Image #{self.image_index}: [{status}] (score: {self.overall_drift_score:.3f})"]
        for detail in self.details:
            lines.append(f"  - {detail}")
        return "\n".join(lines)


class DriftDetector:
    """
    Compares generated images against locked facial parameters.

    Detects when AI-generated images have drifted from the target
    biometric measurements, enabling quality control across batches.

    Usage:
        detector = DriftDetector(target_measurements)
        report = detector.check("generated_image.png")
        if not report.is_within_tolerance:
            print(f"Drift detected: {report}")
    """

    # Default tolerances for each measurement
    DEFAULT_TOLERANCES: Dict[str, float] = {
        "gonial_angle": 5.0,  # ±5 degrees
        "facial_index": 0.1,  # ±0.1 ratio
        "canthal_tilt": 3.0,  # ±3 degrees
        "nasal_tip_rotation": 8.0,  # ±8 degrees
        "zygomatic_prominence": 0.05,  # ±0.05 normalized
        "philtrum_length": 0.05,  # ±0.05 normalized
        "lip_fullness_ratio": 0.15,  # ±0.15 ratio
        "brow_arch_height": 0.03,  # ±0.03 normalized
        "face_symmetry_score": 0.15,  # ±0.15
    }

    def __init__(
        self,
        target: FacialMeasurements,
        tolerances: Optional[Dict[str, float]] = None,
        analyzer: Optional[FaceAnalyzer] = None,
    ) -> None:
        self.target = target
        self.tolerances = {**self.DEFAULT_TOLERANCES, **(tolerances or {})}
        self._analyzer = analyzer or FaceAnalyzer()
        self._owns_analyzer = analyzer is None

    def check(
        self, image_input: Any, image_index: int = 0
    ) -> DriftReport:
        """Check a single image for drift from target measurements."""
        actual = self._analyzer.analyze(image_input)
        return self._compare(actual, image_index)

    def check_batch(
        self, image_inputs: List[Any]
    ) -> List[DriftReport]:
        """Check multiple images for drift. Returns a report for each."""
        reports = []
        for i, img in enumerate(image_inputs):
            try:
                reports.append(self.check(img, image_index=i))
            except ValueError as e:
                reports.append(
                    DriftReport(
                        image_index=i,
                        deviations={},
                        overall_drift_score=1.0,
                        is_within_tolerance=False,
                        details=[f"Analysis failed: {e}"],
                    )
                )
        return reports

    def _compare(
        self, actual: FacialMeasurements, image_index: int
    ) -> DriftReport:
        """Compare actual measurements against target."""
        numeric_fields = [
            "gonial_angle",
            "facial_index",
            "canthal_tilt",
            "nasal_tip_rotation",
            "zygomatic_prominence",
            "philtrum_length",
            "lip_fullness_ratio",
            "brow_arch_height",
            "face_symmetry_score",
        ]

        deviations: Dict[str, float] = {}
        details: List[str] = []
        normalized_drifts: List[float] = []

        for field_name in numeric_fields:
            target_val = getattr(self.target, field_name)
            actual_val = getattr(actual, field_name)
            deviation = abs(actual_val - target_val)
            tolerance = self.tolerances.get(field_name, float("inf"))

            deviations[field_name] = deviation

            if tolerance > 0:
                normalized_drifts.append(deviation / tolerance)

            if deviation > tolerance:
                unit = "°" if field_name in (
                    "gonial_angle", "canthal_tilt", "nasal_tip_rotation"
                ) else ""
                details.append(
                    f"{field_name}: {actual_val:.1f}{unit} vs target "
                    f"{target_val:.1f}{unit} (±{tolerance}{unit} tolerance)"
                )

        # Check categorical drift
        if actual.eye_shape != self.target.eye_shape:
            details.append(
                f"eye_shape: {actual.eye_shape.value} vs target "
                f"{self.target.eye_shape.value}"
            )
            normalized_drifts.append(1.0)

        overall_drift = (
            float(np.mean(normalized_drifts)) if normalized_drifts else 0.0
        )
        is_within = len(details) == 0

        return DriftReport(
            image_index=image_index,
            deviations=deviations,
            overall_drift_score=min(overall_drift, 1.0),
            is_within_tolerance=is_within,
            details=details if details else ["All measurements within tolerance"],
        )

    def close(self) -> None:
        """Release resources."""
        if self._owns_analyzer:
            self._analyzer.close()

    def __enter__(self) -> "DriftDetector":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
