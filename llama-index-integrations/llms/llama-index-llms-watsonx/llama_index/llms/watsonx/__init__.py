import warnings

from llama_index.llms.watsonx.base import WatsonX

__all__ = ["WatsonX"]

warnings.warn(
    (
        "llama_index.llms.watsonx.base.WatsonX class is deprecated in favor of"
        " llama_index.llms.ibm.base.WatsonxLLM . To install"
        " llama-index-llms-ibm run `pip install -U llama-index-llms-ibm`."
    ),
    DeprecationWarning,
)
