#!/usr/bin/env python3
"""
Face Lock headless CLI analysis script.

Bridges the Node.js OpenClaw skill to the actual face_lock_biometric engine.
NOT a reimplementation — imports the real FaceAnalyzer, PromptGenerator, and
DriftDetector so the bot uses the exact same code as the Python package.

Usage:
    python3 run_analysis.py --image photo.jpg --json-output
    python3 run_analysis.py --image photo.jpg --platform reve --json-output
    python3 run_analysis.py --image photo.jpg --mode synthetic --seed 42 --json-output
    python3 run_analysis.py --drift photo.jpg --target-json profile.json --json-output
"""

import argparse
import json
import os
import sys

# Suppress TensorFlow/MediaPipe noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"


def _error(msg: str, code: int = 1) -> None:
    """Print a JSON error and exit."""
    print(json.dumps({"error": msg}))
    sys.exit(code)


def run_analyze(args: argparse.Namespace) -> None:
    """Analyze a real face image and output measurements + prompts."""
    try:
        from llama_index.packs.face_lock_biometric.face_lock import FaceAnalyzer
        from llama_index.packs.face_lock_biometric.prompt_generators import (
            Platform,
            PromptGenerator,
        )
    except ImportError:
        _error(
            "face_lock_biometric package not installed. "
            "Run: pip install llama-index-packs-face-lock-biometric"
        )

    image_path = args.image
    if not os.path.exists(image_path):
        _error(f"Image not found: {image_path}")

    try:
        analyzer = FaceAnalyzer()
    except Exception as e:
        _error(f"MediaPipe initialization failed: {e}")

    try:
        measurements = analyzer.analyze(image_path)
    except ValueError as e:
        _error(str(e))
    except Exception as e:
        _error(f"Analysis failed: {e}")
    finally:
        analyzer.close()

    # Generate prompts
    platform_map = {
        "reve": Platform.REVE,
        "flux": Platform.FLUX,
        "midjourney": Platform.MIDJOURNEY,
        "generic": Platform.GENERIC,
    }
    platform = platform_map.get(args.platform, Platform.GENERIC)

    gen = PromptGenerator(
        platform=platform,
        include_technical=args.include_technical,
    )
    positive, negative = gen.generate(
        measurements,
        additional_context=args.context or "",
    )

    output = {
        "status": "success",
        "mode": "real_face",
        **measurements.to_dict(),
        "positive_prompt": positive,
        "negative_prompt": negative,
        "platform": args.platform,
    }

    if args.json_output:
        print(json.dumps(output))
    else:
        print(f"Gonial Angle: {measurements.gonial_angle:.1f}°")
        print(f"Canthal Tilt: {measurements.canthal_tilt:.1f}°")
        print(f"Facial Index: {measurements.facial_index:.3f}")
        print(f"Eye Shape: {measurements.eye_shape.value}")
        print(f"Fitzpatrick: Type {measurements.fitzpatrick_type.value}")
        print(f"\nPositive Prompt:\n{positive}")
        print(f"\nNegative Prompt:\n{negative}")


def run_synthetic(args: argparse.Namespace) -> None:
    """Generate a synthetic character and output measurements + prompts."""
    try:
        from llama_index.packs.face_lock_biometric.prompt_generators import (
            Platform,
            PromptGenerator,
        )
        from llama_index.packs.face_lock_biometric.workflows import (
            generate_synthetic_measurements,
        )
    except ImportError:
        _error("face_lock_biometric package not installed.")

    seed = args.seed
    overrides = {}
    if args.gonial_angle is not None:
        overrides["gonial_angle"] = args.gonial_angle
    if args.canthal_tilt is not None:
        overrides["canthal_tilt"] = args.canthal_tilt
    if args.facial_index is not None:
        overrides["facial_index"] = args.facial_index

    measurements = generate_synthetic_measurements(
        seed=seed,
        overrides=overrides if overrides else None,
    )

    platform_map = {
        "reve": Platform.REVE,
        "flux": Platform.FLUX,
        "midjourney": Platform.MIDJOURNEY,
        "generic": Platform.GENERIC,
    }
    platform = platform_map.get(args.platform, Platform.GENERIC)

    gen = PromptGenerator(platform=platform, include_technical=args.include_technical)
    positive, negative = gen.generate(
        measurements,
        additional_context=args.context or "",
    )

    output = {
        "status": "success",
        "mode": "synthetic",
        **measurements.to_dict(),
        "positive_prompt": positive,
        "negative_prompt": negative,
        "platform": args.platform,
        "seed": seed,
    }

    if args.json_output:
        print(json.dumps(output))
    else:
        print(f"Synthetic Character (seed={seed})")
        print(f"Gonial Angle: {measurements.gonial_angle:.1f}°")
        print(f"Canthal Tilt: {measurements.canthal_tilt:.1f}°")
        print(f"Facial Index: {measurements.facial_index:.3f}")
        print(f"\nPositive Prompt:\n{positive}")
        print(f"\nNegative Prompt:\n{negative}")


def run_drift(args: argparse.Namespace) -> None:
    """Check a generated image for drift against a target profile."""
    try:
        from llama_index.packs.face_lock_biometric.face_lock import (
            DriftDetector,
            FaceAnalyzer,
            FacialMeasurements,
            EyeShape,
            FitzpatrickScale,
        )
    except ImportError:
        _error("face_lock_biometric package not installed.")

    if not args.drift:
        _error("--drift requires an image path.")
    if not args.target_json:
        _error("--drift requires --target-json with the target profile.")

    image_path = args.drift
    if not os.path.exists(image_path):
        _error(f"Image not found: {image_path}")

    target_path = args.target_json
    if not os.path.exists(target_path):
        _error(f"Target profile not found: {target_path}")

    with open(target_path) as f:
        target_data = json.load(f)

    # Reconstruct measurements from JSON
    m = target_data.get("measurements", target_data)
    target = FacialMeasurements(
        gonial_angle=m.get("gonial_angle", 0),
        facial_index=m.get("facial_index", 0),
        canthal_tilt=m.get("canthal_tilt", 0),
        nasal_tip_rotation=m.get("nasal_tip_rotation", 0),
        zygomatic_prominence=m.get("zygomatic_prominence", 0),
        philtrum_length=m.get("philtrum_length", 0),
        lip_fullness_ratio=m.get("lip_fullness_ratio", 0),
        brow_arch_height=m.get("brow_arch_height", 0),
        face_symmetry_score=m.get("face_symmetry_score", 0),
        eye_shape=EyeShape(m.get("eye_shape", "almond")),
        fitzpatrick_type=FitzpatrickScale(m.get("fitzpatrick_type", 3)),
    )

    analyzer = FaceAnalyzer()
    detector = DriftDetector(target=target, analyzer=analyzer)

    try:
        report = detector.check(image_path, image_index=0)
    except ValueError as e:
        _error(str(e))
    finally:
        analyzer.close()

    output = {
        "status": "success",
        "mode": "drift_check",
        "image": image_path,
        "is_within_tolerance": report.is_within_tolerance,
        "overall_drift_score": round(report.overall_drift_score, 3),
        "deviations": {k: round(v, 3) for k, v in report.deviations.items()},
        "details": report.details,
    }

    if args.json_output:
        print(json.dumps(output))
    else:
        print(str(report))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face Lock CLI — Headless biometric analysis"
    )

    # Mode selection
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a face image for real-face analysis",
    )
    parser.add_argument(
        "--mode", type=str, default="real_face",
        choices=["real_face", "synthetic"],
        help="Workflow mode (default: real_face if --image given, synthetic otherwise)",
    )
    parser.add_argument(
        "--drift", type=str, default=None,
        help="Path to a generated image for drift checking",
    )
    parser.add_argument(
        "--target-json", type=str, default=None,
        help="Path to target profile JSON (for drift checking)",
    )

    # Synthetic mode options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gonial-angle", type=float, default=None)
    parser.add_argument("--canthal-tilt", type=float, default=None)
    parser.add_argument("--facial-index", type=float, default=None)

    # Output options
    parser.add_argument(
        "--platform", type=str, default="generic",
        choices=["reve", "flux", "midjourney", "generic"],
        help="Target AI platform for prompt generation",
    )
    parser.add_argument("--context", type=str, default=None, help="Additional prompt context")
    parser.add_argument("--json-output", action="store_true", help="Output as JSON")
    parser.add_argument("--include-technical", action="store_true", help="Include raw values in prompts")

    args = parser.parse_args()

    # Route to the right handler
    if args.drift:
        run_drift(args)
    elif args.image:
        run_analyze(args)
    elif args.mode == "synthetic" or not args.image:
        run_synthetic(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
