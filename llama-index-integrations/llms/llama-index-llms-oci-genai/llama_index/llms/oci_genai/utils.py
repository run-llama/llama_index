import inspect
import json
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Sequence, Dict, Union, Type, Callable, Optional, List
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.tools import BaseTool
from oci.generative_ai_inference.models import CohereTool


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
    "cohere.command-r-16k": 16000,
    "cohere.command-r-plus": 128000,  # placeholder for future support
    "meta.llama-3-70b-instruct": 8192,
}

OCIGENAI_LLMS = {**COMPLETION_MODELS, **CHAT_MODELS}

STREAMING_MODELS = {
    "cohere.command",
    "cohere.command-light",
    "meta.llama-2-70b-chat",
    "cohere.command-r-16k",
    "cohere.command-r-plus",
    "meta.llama-3-70b-instruct",
}

JSON_TO_PYTHON_TYPES = {
    "string": "str",
    "number": "float",
    "boolean": "bool",
    "integer": "int",
    "array": "List",
    "object": "Dict",
}


def _format_oci_tool_calls(
        tool_calls: Optional[List[Any]] = None,
) -> List[Dict]:
    """
    Formats an OCI GenAI API response into the tool call format used in LlamaIndex.
    """
    if not tool_calls:
        return []

    formatted_tool_calls = []
    for tool_call in tool_calls:
        formatted_tool_calls.append(
            {
                "toolUseId": uuid.uuid4().hex[:],
                "name": tool_call.name,
                "input": json.dumps(tool_call.parameters),
            }
        )
    return formatted_tool_calls



def create_client(auth_type, auth_profile, service_endpoint):
    """OCI Gen AI client.

    Args:
        auth_type (Optional[str]): Authentication type, can be: API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL.
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
            client_kwargs["config"] = oci.config.from_file(profile_name=auth_profile)
            client_kwargs.pop("signer", None)
        elif auth_type == OCIAuthType(2).name:

            def make_security_token_signer(oci_config):  # type: ignore[no-untyped-def]
                pk = oci.signer.load_private_key_from_file(
                    oci_config.get("key_file"), None
                )
                with open(oci_config.get("security_token_file"), encoding="utf-8") as f:
                    st_string = f.read()
                return oci.auth.signers.SecurityTokenSigner(st_string, pk)

            client_kwargs["config"] = oci.config.from_file(profile_name=auth_profile)
            client_kwargs["signer"] = make_security_token_signer(
                oci_config=client_kwargs["config"]
            )
        elif auth_type == OCIAuthType(3).name:
            client_kwargs[
                "signer"
            ] = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        elif auth_type == OCIAuthType(4).name:
            client_kwargs["signer"] = oci.auth.signers.get_resource_principals_signer()
        else:
            raise ValueError(
                f"Please provide valid value to auth_type, {auth_type} is not valid."
            )

        return oci.generative_ai_inference.GenerativeAiInferenceClient(**client_kwargs)

    except ImportError as ex:
        raise ModuleNotFoundError(
            "Could not import oci python package. "
            "Please make sure you have the oci package installed."
        ) from ex
    except Exception as e:
        raise ValueError(
            """Could not authenticate with OCI client. Please check if ~/.oci/config exists.
            If INSTANCE_PRINCIPAL or RESOURCE_PRINCIPAL is used, please check the specified
            auth_profile and auth_type are valid.""",
            e,
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
    def chat_generation_info(self, response: Any) -> Dict[str, Any]:
        ...

    @abstractmethod
    def chat_stream_generation_info(self, event_data: Dict) -> Dict[str, Any]:
        ...

    @abstractmethod
    def messages_to_oci_params(self, messages: Sequence[ChatMessage]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def convert_to_oci_tool(
            self,
            tool: Union[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
    ) -> Dict[str, Any]: ...


class CohereProvider(Provider):
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
        self.oci_tool = models.CohereTool
        self.oci_tool_param = models.CohereParameterDefinition
        self.oci_tool_result = models.CohereToolResult
        self.oci_tool_call = models.CohereToolCall
        self.oci_chat_message = {
            "USER": models.CohereUserMessage,
            "CHATBOT": models.CohereChatBotMessage,
            "SYSTEM": models.CohereSystemMessage,
            "TOOL": models.CohereToolMessage,
        }
        self.chat_api_format = models.BaseChatRequest.API_FORMAT_COHERE

    def completion_response_to_text(self, response: Any) -> str:
        return response.data.inference_response.generated_texts[0].text

    def completion_stream_to_text(self, event_data: Any) -> str:
        return event_data["text"]

    def chat_response_to_text(self, response: Any) -> str:
        return response.data.chat_response.text

    def chat_stream_to_text(self, event_data: Dict) -> str:
        if "text" in event_data and "finishReason" not in event_data:
            return event_data["text"]
        else:
            return ""

    def chat_generation_info(self, response: Any) -> Dict[str, Any]:
        generation_info: Dict[str, Any] = {
            "documents": response.data.chat_response.documents,
            "citations": response.data.chat_response.citations,
            "search_queries": response.data.chat_response.search_queries,
            "is_search_required": response.data.chat_response.is_search_required,
            "finish_reason": response.data.chat_response.finish_reason,
        }
        if response.data.chat_response.tool_calls:
            # Only populate tool_calls when 1) present on the response and
            #  2) has one or more calls.
            generation_info["tool_calls"] = _format_oci_tool_calls(
                response.data.chat_response.tool_calls
            )

        return generation_info

    def chat_stream_generation_info(self, event_data: Dict) -> Dict[str, Any]:
        generation_info: Dict[str, Any] = {
            "documents": event_data.get("documents"),
            "citations": event_data.get("citations"),
            "finish_reason": event_data.get("finishReason"),
        }
        if "toolCalls" in event_data:
            generation_info["tool_calls"] = []
            for tool_call in event_data["toolCalls"]:
                generation_info["tool_calls"].append(
                    {
                        "toolUseId": uuid.uuid4().hex[:],
                        "name": tool_call.name,
                        "input": json.dumps(tool_call.parameters),
                    }
                )

        generation_info = {k: v for k, v in generation_info.items() if v is not None}

        return generation_info

    def messages_to_oci_params(self, messages: Sequence[ChatMessage]) -> Dict[str, Any]:
        role_map = {
            "user": "USER",
            "system": "SYSTEM",
            "assistant": "CHATBOT",
            "tool": "TOOL",
            "function": "TOOL",
            "chatbot": "CHATBOT",
            "model": "CHATBOT"
        }

        oci_chat_history = []

        for msg in messages[:-1]:
            role = role_map[msg.role.value]

            # Handle tool calls for AI/Assistant messages
            if role == "CHATBOT" and "tool_calls" in msg.additional_kwargs:
                tool_calls = []
                for tool_call in msg.additional_kwargs.get("tool_calls", []):
                    validate_tool_call(tool_call)
                    tool_calls.append(self.oci_tool_call(
                        name=tool_call.get("name"),
                        parameters=json.loads(tool_call["input"])
                        if isinstance(tool_call["input"], str)
                        else tool_call["input"]
                    ))

                oci_chat_history.append(
                    self.oci_chat_message[role](
                        message=msg.content if msg.content else " ",
                        tool_calls=tool_calls if tool_calls else None
                    )
                )
            else:
                oci_chat_history.append(
                    self.oci_chat_message[role](
                        message=msg.content
                    )
                )

        # Handling the current chat turn, especially the latest message
        current_chat_turn_messages = []
        for message in messages[::-1]:
            current_chat_turn_messages.append(message)
            if message.role == MessageRole.USER:
                break
        current_chat_turn_messages = current_chat_turn_messages[::-1]

        oci_tool_results = []
        for message in current_chat_turn_messages:
            if message.role == MessageRole.TOOL:
                tool_message = message
                previous_ai_msgs = [
                    message
                    for message in current_chat_turn_messages
                    if message.role == MessageRole.ASSISTANT and "tool_calls" in message.additional_kwargs
                ]
                if previous_ai_msgs:
                    previous_ai_msg = previous_ai_msgs[-1]
                    for li_tool_call in previous_ai_msg.additional_kwargs.get("tool_calls", []):
                        validate_tool_call(li_tool_call)
                        if li_tool_call["toolUseId"] == tool_message.additional_kwargs.get("tool_call_id"):
                            tool_result = self.oci_tool_result()
                            tool_result.call = self.oci_tool_call(
                                name=li_tool_call.get("name"),
                                parameters=json.loads(li_tool_call["input"])
                                if isinstance(li_tool_call["input"], str)
                                else li_tool_call["input"]
                            )
                            tool_result.outputs = [{"output": tool_message.content}]
                            oci_tool_results.append(tool_result)

        if not oci_tool_results:
            oci_tool_results = None

        message_str = "" if oci_tool_results else messages[-1].content

        oci_params = {
            "message": message_str,
            "chat_history": oci_chat_history,
            "tool_results": oci_tool_results,
            "api_format": self.chat_api_format,
        }

        return {k: v for k, v in oci_params.items() if v is not None}

    def convert_to_oci_tool(self,
                            tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
                            ) -> CohereTool:
        """
        Convert a Pydantic class, JSON schema dict, callable, or BaseTool to a CohereTool format for OCI.

        Args:
            tool: The tool to convert, which can be a Pydantic class, a callable, or a JSON schema dictionary.

        Returns:
            A CohereTool representing the tool in the OCI API format.
        """
        if isinstance(tool, BaseTool):
            # Extract tool name and description for BaseTool
            tool_name, tool_description = getattr(tool, "name", None), getattr(tool, "description", None)
            if not tool_name or not tool_description:
                tool_name = getattr(tool.metadata, "name", None)
                if tool_fn := getattr(tool, "fn", None):
                    tool_description = tool_fn.__doc__
                    if not tool_name:
                        tool_name = tool_fn.__name__
                else:
                    tool_description = getattr(tool.metadata, "description", None)
                if not tool_name or not tool_description:
                    raise ValueError(f"Tool {tool} does not have a name or description.")

            return self.oci_tool(
                name=tool_name,
                description=tool_description,
                parameters={
                    p_name: self.oci_tool_param(
                        type=JSON_TO_PYTHON_TYPES.get(p_def.get("type"), p_def.get("type")),
                        description=p_def.get("description", ""),
                        is_required=p_name in tool.metadata.get_parameters_dict().get("required", []))
                    for p_name, p_def in tool.metadata.get_parameters_dict().get("properties", {}).items()
                },
            )

        elif isinstance(tool, dict):
            # Ensure dict-based tools follow a standard schema format
            if not all(k in tool for k in ("title", "description", "properties")):
                raise ValueError(
                    "Unsupported dict type. Tool must be passed in as a BaseTool instance, JSON schema dict, or BaseModel type."
                )
            return self.oci_tool(
                name=tool.get("title"),
                description=tool.get("description"),
                parameters={
                    p_name: self.oci_tool_param(
                        type=JSON_TO_PYTHON_TYPES.get(p_def.get("type"), p_def.get("type")),
                        description=p_def.get("description", ""),
                        is_required=p_name in tool.get("required", [])
                    )
                    for p_name, p_def in tool.get("properties", {}).items()
                },
            )

        elif isinstance(tool, type) and issubclass(tool, BaseModel):
            # Handle Pydantic BaseModel tools
            schema = tool.model_json_schema()
            properties = schema.get("properties", {})
            return self.oci_tool(
                name=schema.get("title", tool.__name__),
                description=schema.get("description", tool.__name__),
                parameters={
                    p_name: self.oci_tool_param(
                        type=JSON_TO_PYTHON_TYPES.get(p_def.get("type"), p_def.get("type")),
                        description=p_def.get("description", ""),
                        is_required=p_name in schema.get("required", [])
                    )
                    for p_name, p_def in properties.items()
                },
            )

        elif callable(tool):
            # Use inspect to extract callable signature and arguments
            signature = inspect.signature(tool)
            parameters = {}
            for param_name, param in signature.parameters.items():
                param_type = param.annotation if param.annotation != inspect._empty else "string"
                param_default = param.default if param.default != inspect._empty else None

                # Convert type to JSON schema type (or leave as default)
                json_type = JSON_TO_PYTHON_TYPES.get(
                    param_type, param_type.__name__ if isinstance(param_type, type) else "string"
                )

                parameters[param_name] = {
                    "type": json_type,
                    "description": f"Parameter: {param_name}",
                    "is_required": param_default is None
                }

            return self.oci_tool(
                name=tool.__name__,
                description=tool.__doc__ or f"Callable function: {tool.__name__}",
                parameters={
                    param_name: self.oci_tool_param(
                        type=param_data["type"],
                        description=param_data["description"],
                        is_required=param_data["is_required"]
                    )
                    for param_name, param_data in parameters.items()
                },
            )

        else:
            raise ValueError(
                f"Unsupported tool type {type(tool)}. Tool must be passed in as a BaseTool instance, JSON schema dict, or BaseModel type."
            )


class MetaProvider(Provider):
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
        self.oci_chat_message = {
            "USER": models.UserMessage,
            "SYSTEM": models.SystemMessage,
            "ASSISTANT": models.AssistantMessage,
        }
        self.oci_chat_message_content = models.TextContent
        self.chat_api_format = models.BaseChatRequest.API_FORMAT_GENERIC

    def completion_response_to_text(self, response: Any) -> str:
        return response.data.inference_response.choices[0].text

    def completion_stream_to_text(self, event_data: Any) -> str:
        return event_data["text"]

    def chat_response_to_text(self, response: Any) -> str:
        return response.data.chat_response.choices[0].message.content[0].text

    def chat_stream_to_text(self, event_data: Dict) -> str:
        if "message" in event_data:
            return event_data["message"]["content"][0]["text"]
        else:
            return ""

    def chat_generation_info(self, response: Any) -> Dict[str, Any]:
        return {
            "finish_reason": response.data.chat_response.choices[0].finish_reason,
            "time_created": str(response.data.chat_response.time_created),
        }

    def chat_stream_generation_info(self, event_data: Dict) -> Dict[str, Any]:
        return {
            "finish_reason": event_data["finishReason"],
        }

    def messages_to_oci_params(self, messages: Sequence[ChatMessage]) -> Dict[str, Any]:
        role_map = {
            "user": "USER",
            "system": "SYSTEM",
            "chatbot": "ASSISTANT",
            "assistant": "ASSISTANT",
        }

        oci_messages = [
            self.oci_chat_message[role_map[msg.role]](
                content=[self.oci_chat_message_content(text=msg.content)],
            )
            for msg in messages
        ]

        return {
            "messages": oci_messages,
            "api_format": self.chat_api_format,
            "top_k": -1,
        }


PROVIDERS = {
    "cohere": CohereProvider(),
    "meta": MetaProvider(),
}


def get_provider(model: str, provider_name: str = None) -> Any:
    if provider_name is None:
        provider_name = model.split(".")[0].lower()

    if provider_name not in PROVIDERS:
        raise ValueError(
            f"Invalid provider derived from model_id: {model} "
            "Please explicitly pass in the supported provider "
            "when using custom endpoint"
        )

    return PROVIDERS[provider_name]


def get_context_size(model: str, context_size: int = None) -> int:
    if context_size is None:
        try:
            return OCIGENAI_LLMS[model]
        except KeyError as e:
            if model.startswith(CUSTOM_ENDPOINT_PREFIX):
                raise ValueError(
                    f"Invalid context size derived from model_id: {model} "
                    "Please explicitly pass in the context size "
                    "when using custom endpoint",
                    e,
                ) from e
            else:
                raise ValueError(
                    f"Invalid model name {model} "
                    "Please double check the following OCI documentation if the model is supported "
                    "https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/generative-ai/pretrained-models.htm#pretrained-models",
                    e,
                ) from e
    else:
        return context_size


def validate_tool_call(tool_call: Dict[str, Any]):
    if (
            "input" not in tool_call
            or "toolUseId" not in tool_call
            or "name" not in tool_call
    ):
        raise ValueError("Invalid tool call.")
