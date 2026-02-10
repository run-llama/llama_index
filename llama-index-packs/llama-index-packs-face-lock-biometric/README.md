# Face Lock: Biometric Character Consistency Pack

Extracts precise facial biometric measurements (gonial angle, canthal tilt, facial index, etc.) from reference images using MediaPipe's 468-landmark face mesh, and converts them to structured prompts for AI image generators (Reve, Flux, Midjourney) with drift detection.

## Installation

```bash
pip install llama-index-packs-face-lock-biometric
```

## Quick Start

### Synthetic Workflow (BIPA-Safe)

```python
from llama_index.packs.face_lock_biometric import FaceLockBiometricPack

pack = FaceLockBiometricPack(mode="synthetic")
result = pack.run(character_name="Hero_001", seed=42, platform="reve")

print(result["positive_prompt"])
print(result["negative_prompt"])
print(result["measurements"])
```

### Real Face Workflow (BIPA-Regulated)

```python
pack = FaceLockBiometricPack(
    mode="real_face",
    image_path="reference.jpg",
    subject_identifier="actor@example.com",
    collector_entity="MyStudio",
)
result = pack.run(character_name="Lead_Actor", platform="flux")
```

## Features

- **Facial Measurement Engine**: Extracts gonial angle, canthal tilt, facial index, nasal tip rotation, zygomatic prominence, eye shape, Fitzpatrick scale
- **Dual-Path Workflow**: BIPA-safe synthetic characters or BIPA-compliant real face analysis
- **Prompt Generation**: Optimized prompts for Reve, Flux, and Midjourney with anti-drift negative prompts
- **Drift Detection**: Batch comparison of generated images against locked parameters
- **Analog Film Processing**: Post-processing with Kodachrome 25, Agfa CNS2, Kodak Vision3 500T presets
- **BIPA Compliance**: Consent recording, data retention enforcement, and audit trails
