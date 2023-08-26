from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union, Mapping, TypedDict
from enum import Enum
from httpx import Timeout


MISSING_API_KEY_ERROR_MESSAGE = """No API key found for Portkey.
Please set either the PORTKEY_API_KEY environment variable or \
pass the api_key prior to initialization of Portkey.
API keys can be found or created at Portkey Dashboard \

Here's how you get it:
1. Visit https://app.portkey.ai/
1. Click on your profile icon on the top left
2. From the dropdown menu, click on "Copy API Key"
"""


DEFAULT_MAX_RETRIES = 2
VERSION = "0.1.0"
DEFAULT_TIMEOUT = 60


class RubeusCacheType(Enum):
    SEMANTIC = "semantic"
    SIMPLE = "simple"


class ProviderTypes(str, Enum):
    """_summary_

    Args:
        Enum (_type_): _description_

    Returns:
        _type_: _description_
    """

    OPENAI = "openai"
    COHERE = "cohere"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure-openai"
    HUGGING_FACE = "huggingface"


class RubeusModes(str, Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """

    FALLBACK = "fallback"
    LOADBALANCE = "loadbalance"
    SINGLE = "single"
    PROXY = "proxy"


class RubeusApiPaths(Enum):
    CHAT_COMPLETION = "/v1/chatComplete"
    COMPLETION = "/v1/complete"


class Options(BaseModel):
    method: str
    url: str
    params: Mapping[str, str] | None
    headers: Mapping[str, str] | None
    max_retries: int
    timeout: Union[float, None]
    # stringified json
    data: Mapping[str, Any]
    # json structure
    json_body: Mapping[str, Any]


class Body(TypedDict):
    timeout: Union[float, None]
    max_retries: int
    provider: ProviderTypes
    model: str
    model_api_key: str
    temperature: float
    top_k: Optional[int]
    top_p: Optional[float]
    stop_sequences: List[str]
    stream: Optional[bool]
    max_tokens: Optional[int]
    trace_id: Optional[str]
    cache_status: Optional[RubeusCacheType]
    cache: Optional[bool]
    metadata: Optional[Dict[str, Any]]
    weight: float


class OverrideParams(BaseModel):
    model: str


class RetrySettings(TypedDict):
    attempts: int
    on_status_codes: list


class ProviderOptions(TypedDict):
    provider: str
    apiKey: str
    weight: float
    override_params: OverrideParams
    retry: RetrySettings


class Config(TypedDict):
    mode: str
    options: List[ProviderOptions]


class Message(BaseModel):
    role: str
    content: str


class Function(BaseModel):
    name: str
    description: str
    parameters: str


class Params(BaseModel):
    messages: List[Message]
    prompt: Union[str, List[str]]
    max_tokens: int
    model: str
    functions: List[Function]
    function_call: Union[None, str, Function]
    temperature: int
    top_p: int
    n: int
    stream: bool
    logprobs: int
    echo: bool
    stop: Union[str, List[str]]
    presence_penalty: int
    frequency_penalty: int
    best_of: int
    logit_bias: Dict[str, int]
    user: str


class RequestData(BaseModel):
    config: Config
    params: Params


def remove_empty_values(data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(data, dict):
        cleaned_dict = {}
        for key, value in data.items():
            if value is not None and value != "":
                cleaned_value = remove_empty_values(value)
                if cleaned_value is not None and cleaned_value != "":
                    cleaned_dict[key] = cleaned_value
        return cleaned_dict
    elif isinstance(data, list):
        cleaned_list = []
        
        for item in data:
            cleaned_item = remove_empty_values(item)
            if cleaned_item is not None and cleaned_item != "":
                cleaned_list.append(cleaned_item)
        return cleaned_list
    else:
        return data


class ProviderBase:
    def __init__(
        self,
        *,
        prompt: str = "",
        messages: Message = [],
        timeout: Union[float, Timeout, None] = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        provider: Optional[ProviderTypes] = ProviderTypes.OPENAI,
        model: str = "gpt-3.5-turbo",
        model_api_key: Optional[str] = None,
        temperature: float = 0.1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_sequences: List[str] = [],
        # stream: Optional[bool] = False,
        max_tokens: Optional[int] = None,
        # trace_id: Optional[str] = "",
        # cache_status: Optional[RubeusCacheType] = "",
        # cache: Optional[bool] = False,
        metadata: Optional[Dict[str, Any]] = None,
        weight: Optional[float] = 1.0,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.provider = provider
        self.model = model
        self.model_api_key = model_api_key
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.stop_sequences = stop_sequences
        self.max_tokens = max_tokens
        self.metadata = metadata or {}
        self.weight = weight
        self.prompt = prompt
        self.messages = messages

    def json(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "messages": self.messages,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "provider": self.provider,
            "model": self.model,
            "model_api_key": self.model_api_key,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": self.stop_sequences,
            "max_tokens": self.max_tokens,
            "metadata": self.metadata,
            "weight": self.weight,
        }
