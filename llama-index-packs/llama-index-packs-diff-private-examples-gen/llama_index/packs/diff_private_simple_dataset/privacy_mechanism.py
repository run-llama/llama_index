from enum import Enum


class PrivacyMechanism(Enum, str):
    """Enum for available privacy mechanism."""

    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
