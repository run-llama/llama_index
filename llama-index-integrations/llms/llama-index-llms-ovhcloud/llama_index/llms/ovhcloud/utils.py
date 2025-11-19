from llama_index.core.bridge.pydantic import BaseModel

# OVHcloud AI Endpoints supported models
# This list can be updated based on the catalog at:
# https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/
SUPPORTED_MODEL_SLUGS = [
    # Add model slugs here as they become available
    # Example format: "model-name"
]


class Model(BaseModel):
    """
    Model information for OVHcloud AI Endpoints models.

    Args:
        id: unique identifier for the model, passed as model parameter for requests
        model_type: API type (defaults to "chat")
        client: client name

    """

    id: str
    model_type: str = "chat"
    client: str = "OVHcloud"

    def __hash__(self) -> int:
        return hash(self.id)
