from enum import Enum
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Union,
    Mapping,
    Literal,
)
from pydantic import BaseModel


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


RubeusCacheLiteral = Literal["semantic", "simple"]


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


ProviderTypesLiteral = Literal[
    "openai", "cohere", "anthropic", "azure-openai", "huggingface"
]


class RubeusModes(str, Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """

    FALLBACK = "fallback"
    LOADBALANCE = "loadbalance"
    SINGLE = "single"
    PROXY = "proxy"


RubeusModesLiteral = Literal["fallback", "loadbalance", "single", "proxy"]


class RubeusApiPaths(Enum):
    CHAT_COMPLETION = "/v1/chatComplete"
    COMPLETION = "/v1/complete"


class Options(BaseModel):
    method: str
    url: str
    params: Optional[Mapping[str, str]] = None
    headers: Optional[Mapping[str, str]] = None
    max_retries: Optional[int] = None
    timeout: Optional[float] = None
    # stringified json
    data: Optional[Mapping[str, Any]] = None
    # json structure
    json_body: Optional[Mapping[str, Any]] = None


class OverrideParams(BaseModel):
    model: str


class RetrySettings(BaseModel):
    attempts: int
    on_status_codes: list


class ProviderOptions(BaseModel):
    provider: Optional[str]
    apiKey: Optional[str]
    weight: Optional[float]
    override_params: Optional[OverrideParams]
    retry: Optional[RetrySettings]


class Config(BaseModel):
    mode: str
    options: List[ProviderOptions]


class Message(BaseModel):
    role: str
    content: str


class Function(BaseModel):
    name: str
    description: str
    parameters: str


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

        for item in data:  # type: ignore
            cleaned_item = remove_empty_values(item)
            if cleaned_item is not None and cleaned_item != "":
                cleaned_list.append(cleaned_item)
        return cleaned_list  # type: ignore
    else:
        return data


class LLMBase(BaseModel):
    """
    provider (Optional[ProviderTypes]): The LLM provider to be used for the Portkey
    integration.
        Eg: openai, anthropic etc.
        NOTE: Check the ProviderTypes to see the supported list of LLMs.
    model (str): The name of the language model to use (default: "gpt-3.5-turbo").
    temperature (float): The temperature parameter for text generation (default: 0.1).
    max_tokens (Optional[int]): The maximum number of tokens in the generated text.
    max_retries (int): The maximum number of retries for failed requests (default: 5).
    trace_id (Optional[str]): A unique identifier for tracing requests.
    cache_status (Optional[RubeusCacheType]): The type of cache to use (default: "").
        If cache_status is set, then cache is automatically set to True
    cache (Optional[bool]): Whether to use caching (default: False).
    metadata (Optional[Dict[str, Any]]): Metadata associated with the request
    (default: {}).
    weight (Optional[float]): The weight of the LLM in the ensemble (default: 1.0).
    """

    prompt: Optional[str] = None
    messages: Optional[List[Message]] = None
    provider: Union[ProviderTypes, ProviderTypesLiteral]
    model: str
    model_api_key: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_retries: Optional[int] = None
    trace_id: Optional[str] = None
    cache_status: Optional[RubeusCacheType] = None
    cache: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    weight: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    timeout: Union[float, None] = None
    retry_settings: Optional[RetrySettings] = None
    # functions: Optional[List[Function]]
    # function_call: Optional[Union[None, str, Function]]
    # n: Optional[int]
    # logprobs: Optional[int]
    # echo: Optional[bool]
    # stop: Optional[Union[str, List[str]]]
    # presence_penalty: Optional[int]
    # frequency_penalty: Optional[int]
    # best_of: Optional[int]
    # logit_bias: Optional[Dict[str, int]]
    # user: Optional[str]

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)


class Body(LLMBase):
    pass


class Params(LLMBase):
    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)


class RequestData(BaseModel):
    config: Config
    params: Params


class RubeusResponse(BaseModel):
    model: str
    choices: List[Any]
    raw_body: Dict[str, Any]
