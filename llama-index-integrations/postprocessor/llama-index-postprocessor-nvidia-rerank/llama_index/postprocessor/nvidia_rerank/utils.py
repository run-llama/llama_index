from dataclasses import dataclass
from typing import Optional
import warnings

BASE_URL = (
    "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking"
)
DEFAULT_MODEL = "nvidia/nv-rerankqa-mistral-4b-v3"


@dataclass
class Model:
    """
    Model information.

    id: unique identifier for the model, passed as model parameter for requests
    model_type: API type (ranking)
    client: client name, e.g. NVIDIARerank
    endpoint: custom endpoint for the model
    aliases: list of aliases for the model

    All aliases are deprecated and will trigger a warning when used.
    """

    id: str
    model_type: Optional[str] = "ranking"
    client: str = "NVIDIARerank"
    endpoint: Optional[str] = None
    aliases: Optional[list] = None
    base_model: Optional[str] = None
    supports_tools: Optional[bool] = False
    supports_structured_output: Optional[bool] = False

    def __hash__(self) -> int:
        return hash(self.id)

    def validate(self):
        if self.client:
            supported = {"NVIDIARerank": ("ranking",)}
            model_type = self.model_type
            if model_type not in supported[self.client]:
                err_msg = (
                    f"Model type '{model_type}' not supported by client '{self.client}'"
                )
                raise ValueError(err_msg)

        return hash(self.id)


RANKING_MODEL_TABLE = {
    "nv-rerank-qa-mistral-4b:1": Model(
        id="nv-rerank-qa-mistral-4b:1",
        model_type="ranking",
        client="NVIDIARerank",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
        aliases=["ai-rerank-qa-mistral-4b"],
    ),
    "nvidia/nv-rerankqa-mistral-4b-v3": Model(
        id="nvidia/nv-rerankqa-mistral-4b-v3",
        model_type="ranking",
        client="NVIDIARerank",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking",
    ),
    "nvidia/llama-3.2-nv-rerankqa-1b-v1": Model(
        id="nvidia/llama-3.2-nv-rerankqa-1b-v1",
        model_type="ranking",
        client="NVIDIARerank",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v1/reranking",
    ),
    "nvidia/llama-3.2-nv-rerankqa-1b-v2": Model(
        id="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        model_type="ranking",
        client="NVIDIARerank",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
    ),
}


def lookup_model(name: str) -> Optional[Model]:
    """
    Lookup a model by name, using only the table of known models.
    The name is either:
        - directly in the table
        - an alias in the table
        - not found (None)
    Callers can check to see if the name was an alias by
    comparing the result's id field to the name they provided.
    """
    if not (model := RANKING_MODEL_TABLE.get(name)):
        for mdl in RANKING_MODEL_TABLE.values():
            if mdl.aliases and name in mdl.aliases:
                model = mdl
                break
    return model


def determine_model(name: str) -> Optional[Model]:
    """
    Determine the model to use based on a name, using
    only the table of known models.

    Raise a warning if the model is found to be
    an alias of a known model.

    If the model is not found, return None.
    """
    if model := lookup_model(name):
        # all aliases are deprecated
        if model.id != name:
            warn_msg = f"Model {name} is deprecated. Using {model.id} instead."
            warnings.warn(warn_msg, UserWarning, stacklevel=1)
    return model


KNOWN_URLS = [
    BASE_URL,
    "https://ai.api.nvidia.com/v1/retrieval/snowflake/arctic-embed-l",
]
