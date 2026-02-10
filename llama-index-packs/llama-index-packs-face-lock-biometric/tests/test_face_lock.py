"""Tests for the Face Lock biometric measurement engine."""

import json
import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llama_index.packs.face_lock_biometric.face_lock import (
    DriftDetector,
    DriftReport,
    EyeShape,
    FacialMeasurements,
    FitzpatrickScale,
    _angle_between_points,
    _distance,
    _midpoint,
)


class TestGeometryHelpers:
    """Test low-level geometry functions."""

    def test_distance_same_point(self) -> None:
        p = np.array([1.0, 2.0])
        assert _distance(p, p) == pytest.approx(0.0)

    def test_distance_known(self) -> None:
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])
        assert _distance(p1, p2) == pytest.approx(5.0)

    def test_midpoint(self) -> None:
        p1 = np.array([0.0, 0.0])
        p2 = np.array([4.0, 6.0])
        mid = _midpoint(p1, p2)
        assert mid[0] == pytest.approx(2.0)
        assert mid[1] == pytest.approx(3.0)

    def test_angle_right_angle(self) -> None:
        p1 = np.array([1.0, 0.0])
        vertex = np.array([0.0, 0.0])
        p2 = np.array([0.0, 1.0])
        angle = _angle_between_points(p1, vertex, p2)
        assert angle == pytest.approx(90.0, abs=0.1)

    def test_angle_straight_line(self) -> None:
        p1 = np.array([-1.0, 0.0])
        vertex = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        angle = _angle_between_points(p1, vertex, p2)
        assert angle == pytest.approx(180.0, abs=0.1)

    def test_angle_acute(self) -> None:
        p1 = np.array([1.0, 0.0])
        vertex = np.array([0.0, 0.0])
        p2 = np.array([1.0, 1.0])
        angle = _angle_between_points(p1, vertex, p2)
        assert angle == pytest.approx(45.0, abs=0.1)


class TestFacialMeasurements:
    """Test the FacialMeasurements dataclass."""

    def test_default_values(self) -> None:
        m = FacialMeasurements()
        assert m.gonial_angle == 0.0
        assert m.facial_index == 0.0
        assert m.eye_shape == EyeShape.ALMOND
        assert m.fitzpatrick_type == FitzpatrickScale.TYPE_III
        assert m.raw_landmarks is None

    def test_to_dict_excludes_raw_landmarks(self) -> None:
        m = FacialMeasurements(
            gonial_angle=128.5,
            facial_index=1.32,
            raw_landmarks=[(0.1, 0.2, 0.3)],
        )
        d = m.to_dict()
        assert "raw_landmarks" not in d
        assert d["gonial_angle"] == 128.5
        assert d["facial_index"] == 1.32

    def test_to_dict_rounds_values(self) -> None:
        m = FacialMeasurements(
            gonial_angle=128.456789,
            facial_index=1.3267891,
            canthal_tilt=2.345678,
        )
        d = m.to_dict()
        assert d["gonial_angle"] == 128.5
        assert d["facial_index"] == 1.327
        assert d["canthal_tilt"] == 2.3

    def test_to_dict_serializes_enums(self) -> None:
        m = FacialMeasurements(
            eye_shape=EyeShape.HOODED,
            fitzpatrick_type=FitzpatrickScale.TYPE_V,
        )
        d = m.to_dict()
        assert d["eye_shape"] == "hooded"
        assert d["fitzpatrick_type"] == 5

    def test_custom_measurements(self) -> None:
        m = FacialMeasurements(
            gonial_angle=125.0,
            facial_index=1.35,
            canthal_tilt=3.0,
            nasal_tip_rotation=98.0,
            zygomatic_prominence=0.15,
            eye_shape=EyeShape.UPTURNED,
            fitzpatrick_type=FitzpatrickScale.TYPE_IV,
        )
        assert m.gonial_angle == 125.0
        assert m.eye_shape == EyeShape.UPTURNED


class TestDriftDetector:
    """Test drift detection logic."""

    def _make_target(self) -> FacialMeasurements:
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

    def test_identical_measurements_pass(self) -> None:
        target = self._make_target()

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = self._make_target()
        detector = DriftDetector(target=target, analyzer=mock_analyzer)

        report = detector.check("fake_image.jpg", image_index=0)
        assert report.is_within_tolerance
        assert report.overall_drift_score == pytest.approx(0.0, abs=0.01)
        assert report.image_index == 0

    def test_drifted_jaw_detected(self) -> None:
        target = self._make_target()

        drifted = self._make_target()
        drifted.gonial_angle = 140.0  # 12° drift, tolerance is 5°

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = drifted
        detector = DriftDetector(target=target, analyzer=mock_analyzer)

        report = detector.check("fake_image.jpg", image_index=7)
        assert not report.is_within_tolerance
        assert report.image_index == 7
        assert any("gonial_angle" in d for d in report.details)

    def test_drifted_eye_shape_detected(self) -> None:
        target = self._make_target()

        drifted = self._make_target()
        drifted.eye_shape = EyeShape.ROUND  # Categorical drift

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = drifted
        detector = DriftDetector(target=target, analyzer=mock_analyzer)

        report = detector.check("fake_image.jpg")
        assert not report.is_within_tolerance
        assert any("eye_shape" in d for d in report.details)

    def test_within_tolerance(self) -> None:
        target = self._make_target()

        # Slight drift within tolerance
        slight = self._make_target()
        slight.gonial_angle = 130.0  # 2° drift, tolerance is 5°
        slight.canthal_tilt = 3.5  # 1.5° drift, tolerance is 3°

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = slight
        detector = DriftDetector(target=target, analyzer=mock_analyzer)

        report = detector.check("fake_image.jpg")
        assert report.is_within_tolerance

    def test_custom_tolerances(self) -> None:
        target = self._make_target()

        drifted = self._make_target()
        drifted.gonial_angle = 130.0  # 2° drift, tolerance now 1°

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = drifted
        # Very tight tolerance
        detector = DriftDetector(
            target=target,
            tolerances={"gonial_angle": 1.0},
            analyzer=mock_analyzer,
        )

        report = detector.check("fake_image.jpg")
        assert not report.is_within_tolerance

    def test_batch_check(self) -> None:
        target = self._make_target()

        mock_analyzer = MagicMock()
        # First image passes, second drifts
        pass_m = self._make_target()
        drift_m = self._make_target()
        drift_m.gonial_angle = 145.0

        mock_analyzer.analyze.side_effect = [pass_m, drift_m]
        detector = DriftDetector(target=target, analyzer=mock_analyzer)

        reports = detector.check_batch(["img1.jpg", "img2.jpg"])
        assert len(reports) == 2
        assert reports[0].is_within_tolerance
        assert not reports[1].is_within_tolerance

    def test_batch_handles_failures(self) -> None:
        target = self._make_target()

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = ValueError("No face detected")
        detector = DriftDetector(target=target, analyzer=mock_analyzer)

        reports = detector.check_batch(["bad_img.jpg"])
        assert len(reports) == 1
        assert not reports[0].is_within_tolerance
        assert reports[0].overall_drift_score == 1.0

    def test_drift_report_str(self) -> None:
        report = DriftReport(
            image_index=3,
            deviations={"gonial_angle": 8.0},
            overall_drift_score=0.5,
            is_within_tolerance=False,
            details=["gonial_angle: 136.0° vs target 128.0° (±5.0° tolerance)"],
        )
        s = str(report)
        assert "Image #3" in s
        assert "DRIFT DETECTED" in s
        assert "gonial_angle" in s

    def test_drift_report_str_pass(self) -> None:
        report = DriftReport(
            image_index=0,
            deviations={},
            overall_drift_score=0.0,
            is_within_tolerance=True,
            details=["All measurements within tolerance"],
        )
        s = str(report)
        assert "PASS" in s


class TestEyeShape:
    """Test eye shape enum."""

    def test_values(self) -> None:
        assert EyeShape.ALMOND.value == "almond"
        assert EyeShape.ROUND.value == "round"
        assert EyeShape.HOODED.value == "hooded"
        assert EyeShape.MONOLID.value == "monolid"
        assert EyeShape.DOWNTURNED.value == "downturned"
        assert EyeShape.UPTURNED.value == "upturned"


class TestFitzpatrickScale:
    """Test Fitzpatrick skin type enum."""

    def test_values(self) -> None:
        assert FitzpatrickScale.TYPE_I.value == 1
        assert FitzpatrickScale.TYPE_VI.value == 6
        assert len(FitzpatrickScale) == 6
