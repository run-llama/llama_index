import logging

from llama_index.llms.text_generation_inference.base import (
    TextGenerationInference,
)

logger = logging.getLogger(__name__)

logger.warning("""
===============================================================================
                         ⚠️ DEPRECATION WARNING ⚠️
===============================================================================
The llama-index-llms-text-generation-inference package is NO LONGER MAINTAINED!
Please use HuggingFaceInferenceAPI instead, which can call either TGI servers
or huggingface inference API.
To install: pip install llama-index-llms-huggingface-api
===============================================================================
""".strip())

__all__ = ["TextGenerationInference"]
