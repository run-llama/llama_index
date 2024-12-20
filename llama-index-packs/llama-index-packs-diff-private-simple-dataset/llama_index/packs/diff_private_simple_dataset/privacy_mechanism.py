from enum import Enum


class PrivacyMechanism(str, Enum):
    """Enum for available privacy mechanism."""

    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"
