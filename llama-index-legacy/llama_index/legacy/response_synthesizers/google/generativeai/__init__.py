from llama_index.legacy.vector_stores.google.generativeai import set_google_config

from .base import (
    GoogleTextSynthesizer,
    SynthesizedResponse,
)

__all__ = [
    "GoogleTextSynthesizer",
    "set_google_config",
    "SynthesizedResponse",
]
