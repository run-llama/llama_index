from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Dict, List
from llama_index.core.base.llms.types import ChatMessage

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

CHAT_MODELS = {
    "meta.llama-2-70b-chat": 4096,
    "cohere.command-r": 128000,
}
  
OCIGENAI_LLMS = {**COMPLETION_MODELS, **CHAT_MODELS}

STREAMING_MODELS = {
    "cohere.command",
    "cohere.command-light",
    "meta.llama-2-70b-chat",
    "cohere.command-r"
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


def get_completion_generator() -> Any:
    try:
        from oci.generative_ai_inference import models

    except ImportError as ex:
        raise ModuleNotFoundError(
            "Could not import oci python package. "
            "Please make sure you have the oci package installed."
        ) from ex

    return models.GenerateTextDetails


def get_chat_generator() -> Any:
    try:
        from oci.generative_ai_inference import models

    except ImportError as ex:
        raise ModuleNotFoundError(
            "Could not import oci python package. "
            "Please make sure you have the oci package installed."
        ) from ex

    return models.ChatDetails


class Provider(ABC):
    @property
    @abstractmethod
    def stop_sequence_key(self) -> str:
        ...

    @abstractmethod
    def completion_response_to_text(self, response: Any) -> str:
        ...

    @abstractmethod
    def completion_stream_to_text(self, response: Any) -> str:
        ...

    @abstractmethod
    def chat_response_to_text(self, response: Any) -> str:
        ...

    @abstractmethod
    def chat_stream_to_text(self, event_data: Dict) -> str:
        ...

    @abstractmethod
    def messages_to_oci_params(self, messages: Sequence[ChatMessage]) -> Dict[str, Any]:
        ...

    

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

        self.oci_completion_request = models.CohereLlmInferenceRequest
        self.oci_chat_request = models.CohereChatRequest
        self.oci_chat_message = models.CohereMessage
        self.chat_api_format = models.BaseChatRequest.API_FORMAT_COHERE

    def completion_response_to_text(self, response: Any) -> str:
        return response.data.inference_response.generated_texts[0].text

    def completion_stream_to_text(self, event_data: Any) -> str:
        return event_data["text"]

    def chat_response_to_text(self, response: Any) -> str:
        return response.data.chat_response.text

    def chat_stream_to_text(self, event_data: Dict) -> str:
        return event_data['text']

    def messages_to_oci_params(self, messages: Sequence[ChatMessage]) -> Dict[str, Any]:

        role_map = {
            "user": "USER",
            "system": "SYSTEM",
            "chatbot": "CHATBOT",
            "assistant": "CHATBOT",
            "model": "SYSTEM",
            "function": "SYSTEM",
            "tool": "SYSTEM",
        }
        
        oci_chat_history = [self.oci_chat_message(role=role_map[msg.role.value], message=msg.content) for msg in messages[:-1]]
        oci_params = {
            "message": messages[-1].content,
            "chat_history": oci_chat_history,
            "api_format": self.chat_api_format
        }
    
        return oci_params
    
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

        self.oci_completion_request = models.LlamaLlmInferenceRequest
        self.oci_chat_request = models.GenericChatRequest
        self.oci_chat_message = models.Message
        self.oci_chat_message_content = models.TextContent
        self.chat_api_format = models.BaseChatRequest.API_FORMAT_GENERIC
        
    def completion_response_to_text(self, response: Any) -> str:
        return response.data.inference_response.choices[0].text

    def completion_stream_to_text(self, event_data: Any) -> str:
        return event_data["text"]

    def chat_response_to_text(self, response: Any) -> str:
        return response.data.chat_response.choices[0].message.content[0].text

    def chat_stream_to_text(self, event_data: Dict) -> str:
        return event_data["message"]['content'][0]['text']

    def messages_to_oci_params(self, messages: Sequence[ChatMessage]) -> Dict[str, Any]:

        oci_messages = [self.oci_chat_message(role=msg.role.value, content=[self.oci_chat_message_content(text=msg.content)]) for msg in messages]
        oci_params = {
            "messages": oci_messages,
            "api_format": self.chat_api_format,
            "top_k": -1
        }
    
        return oci_params
        

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


