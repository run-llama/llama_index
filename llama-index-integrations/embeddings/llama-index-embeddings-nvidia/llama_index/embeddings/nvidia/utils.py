from dataclasses import dataclass
from typing import Optional
import warnings

# integrate.api.nvidia.com is the default url for most models, any
# bespoke endpoints will need to be added to the MODEL_ENDPOINT_MAP
BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nv-embedqa-e5-v5"


@dataclass
class Model:
    """
    Model information.

    id: unique identifier for the model, passed as model parameter for requests
    model_type: API type (chat, vlm, embedding, ranking, completions)
    client: client name, e.g. NvidiaGenerator, NVIDIAEmbeddings,
            NVIDIARerank, NvidiaTextEmbedder, NvidiaDocumentEmbedder
    endpoint: custom endpoint for the model
    aliases: list of aliases for the model

    All aliases are deprecated and will trigger a warning when used.
    """

    id: str
    model_type: Optional[str] = "embedding"
    client: str = "NVIDIAEmbedding"
    endpoint: Optional[str] = None
    aliases: Optional[list] = None
    base_model: Optional[str] = None
    supports_tools: Optional[bool] = False
    supports_structured_output: Optional[bool] = False

    def __hash__(self) -> int:
        return hash(self.id)

    def validate(self):
        if self.client:
            supported = {"NVIDIAEmbedding": ("embedding",)}
            model_type = self.model_type
            if model_type not in supported[self.client]:
                err_msg = (
                    f"Model type '{model_type}' not supported by client '{self.client}'"
                )
                raise ValueError(err_msg)

        return hash(self.id)


# because EMBEDDING_MODEL_TABLE is used to construct KNOWN_URLS, we need to
# include at least one model w/ https://integrate.api.nvidia.com/v1
EMBEDDING_MODEL_TABLE = {
    "snowflake/arctic-embed-l": Model(
        id="snowflake/arctic-embed-l",
        model_type="embedding",
        aliases=["ai-arctic-embed-l"],
    ),
    "NV-Embed-QA": Model(
        id="NV-Embed-QA",
        model_type="embedding",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia",
        aliases=[
            "ai-embed-qa-4",
            "playground_nvolveqa_40k",
            "nvolveqa_40k",
        ],
    ),
    "nvidia/nv-embed-v1": Model(
        id="nvidia/nv-embed-v1",
        model_type="embedding",
        aliases=["ai-nv-embed-v1"],
    ),
    "nvidia/nv-embedqa-mistral-7b-v2": Model(
        id="nvidia/nv-embedqa-mistral-7b-v2",
        model_type="embedding",
    ),
    "nvidia/nv-embedqa-e5-v5": Model(
        id="nvidia/nv-embedqa-e5-v5",
        model_type="embedding",
    ),
    "baai/bge-m3": Model(
        id="baai/bge-m3",
        model_type="embedding",
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
    if not (model := EMBEDDING_MODEL_TABLE.get(name)):
        for mdl in EMBEDDING_MODEL_TABLE.values():
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
