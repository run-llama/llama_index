from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Dict
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.ocigenai.llama_utils import (
    messages_to_prompt as messages_to_llama_prompt,
    completion_to_prompt as completion_to_llama_prompt,
)

class OCIAuthType(Enum):
    """OCI authentication types as enumerator."""
    API_KEY = 1
    SECURITY_TOKEN = 2
    INSTANCE_PRINCIPAL = 3
    RESOURCE_PRINCIPAL = 4

CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"

COMPLETION_MODELS = {
    "cohere.command": 4096,
    "cohere.command-light": 4096,
    "meta.llama-2-70b-chat": 4096,
}

CHAT_ONLY_MODELS = {}
  
OCIGENAI_LLMS = {**COMPLETION_MODELS, **CHAT_ONLY_MODELS}

STREAMING_MODELS = {
    "cohere.command",
    "meta.llama-2-70b-chat"
}

def create_client(auth_type, auth_profile, service_endpoint):
    """OCI Gen AI client
    
    Args:
        auth_type (Optional[str]): Authentication type, can be: API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPLE, RESOURCE_PRINCIPLE.
                                    If not specified, API_KEY will be used

        auth_profile (Optional[str]): The name of the profile in ~/.oci/config. If not specified , DEFAULT will be used

        service_endpoint (str): service endpoint url, e.g., "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    """
    try:
        import oci

        client_kwargs = {
            "config": {},
            "signer": None,
            "service_endpoint": service_endpoint,
            "retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY,
            "timeout": (10, 240),  # default timeout config for OCI Gen AI service
        }

        if auth_type == OCIAuthType(1).name:
            client_kwargs["config"] = oci.config.from_file(
                profile_name=auth_profile
            )
            client_kwargs.pop("signer", None)
        elif auth_type == OCIAuthType(2).name:

            def make_security_token_signer(oci_config):  # type: ignore[no-untyped-def]
                pk = oci.signer.load_private_key_from_file(
                    oci_config.get("key_file"), None
                )
                with open(
                    oci_config.get("security_token_file"), encoding="utf-8"
                ) as f:
                    st_string = f.read()
                return oci.auth.signers.SecurityTokenSigner(st_string, pk)

            client_kwargs["config"] = oci.config.from_file(
                profile_name=auth_profile
            )
            client_kwargs["signer"] = make_security_token_signer(
                oci_config=client_kwargs["config"]
            )
        elif auth_type == OCIAuthType(3).name:
            client_kwargs[
                "signer"
            ] = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        elif "auth_type" == OCIAuthType(4).name:
            client_kwargs[
                "signer"
            ] = oci.auth.signers.get_resource_principals_signer()
        else:
            raise ValueError("Please provide valid value to auth_type")
  
        return oci.generative_ai_inference.GenerativeAiInferenceClient(**client_kwargs)
            
    except ImportError as ex:
        raise ModuleNotFoundError(
            "Could not import oci python package. "
            "Please make sure you have the oci package installed."
        ) from ex
    except Exception as e:
        raise ValueError(
            "Could not authenticate with OCI client. "
            "Please check if ~/.oci/config exists. "
            "If INSTANCE_PRINCIPLE or RESOURCE_PRINCIPLE is used, "
            "Please check the specified "
            "auth_profile and auth_type are valid."
        ) from e
    

def get_serving_mode(model_id: str) -> Any:
    try:
        from oci.generative_ai_inference import models

    except ImportError as ex:
        raise ModuleNotFoundError(
            "Could not import oci python package. "
            "Please make sure you have the oci package installed."
        ) from ex

    if model_id.startswith(CUSTOM_ENDPOINT_PREFIX):
        serving_mode = models.DedicatedServingMode(endpoint_id=model_id)
    else:
        serving_mode = models.OnDemandServingMode(model_id=model_id)

    return serving_mode


def get_request_generator() -> Any:
    try:
        from oci.generative_ai_inference import models

    except ImportError as ex:
        raise ModuleNotFoundError(
            "Could not import oci python package. "
            "Please make sure you have the oci package installed."
        ) from ex

    return models.GenerateTextDetails


class Provider(ABC):
    @property
    @abstractmethod
    def stop_sequence_key(self) -> str:
        ...

    @abstractmethod
    def get_text_from_response(self, response: dict) -> str:
        ...

    messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None
    completion_to_prompt: Optional[Callable[[str], str]] = None

class CohereProvider(Provider):
    stop_sequence_key = "stop_sequences"

    def __init__(self) -> None:
        try:
            from oci.generative_ai_inference import models
            
        except ImportError as ex:
            raise ModuleNotFoundError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex

        self.oci_llm_request = models.CohereLlmInferenceRequest

    def get_text_from_response(self, response: Any) -> str:
        return response.data.inference_response.generated_texts[0].text
    
class MetaProvider(Provider):
    stop_sequence_key = "stop"

    def __init__(self) -> None:
        try:
            from oci.generative_ai_inference import models
            
        except ImportError as ex:
            raise ModuleNotFoundError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex

        self.oci_llm_request = models.LlamaLlmInferenceRequest
        self.messages_to_prompt = messages_to_llama_prompt
        self.completion_to_prompt = completion_to_llama_prompt

    def get_text_from_response(self, response: Any) -> str:
        return response.data.inference_response.choices[0].text
        

PROVIDERS = {
    "cohere": CohereProvider(),
    "meta": MetaProvider(),
}

def get_provider(model: str, provider_name: str=None) -> str:
    
    if provider_name is None:
        provider_name = model.split(".")[0].lower()

    if provider_name not in PROVIDERS:
        raise ValueError(
            f"Invalid provider derived from model_id: {model} "
            "Please explicitly pass in the supported provider "
            "when using custom endpoint"
        )
    
    return PROVIDERS[provider_name]


