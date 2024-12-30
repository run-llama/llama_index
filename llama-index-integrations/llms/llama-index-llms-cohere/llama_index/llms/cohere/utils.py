from collections import Counter
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import re

from pydantic import BaseModel

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage, MessageRole

from llama_index.core.prompts.chat_prompts import TEXT_QA_SYSTEM_PROMPT
from llama_index.core.tools.types import BaseTool
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from cohere.types import (
    Tool,
    ToolParameterDefinitionsValue,
)


COMMAND_MODELS = {
    "command-r": 128000,
    "command-r-plus": 128000,
    "command": 4096,
    "command-nightly": 4096,
    "command-light": 4096,
    "command-light-nightly": 4096,
    "c4ai-aya-expanse-32b": 128000,
    "c4ai-aya-expanse-8b": 128000,
}

GENERATION_MODELS = {"base": 2048, "base-light": 2048}

REPRESENTATION_MODELS = {
    "embed-english-light-v2.0": 512,
    "embed-english-v2.0": 512,
    "embed-multilingual-v2.0": 256,
}

FUNCTION_CALLING_MODELS = {"command-r", "command-r-plus"}
ALL_AVAILABLE_MODELS = {**COMMAND_MODELS, **GENERATION_MODELS, **REPRESENTATION_MODELS}
CHAT_MODELS = {**COMMAND_MODELS}

JSON_TO_PYTHON_TYPES = {
    "string": "str",
    "number": "float",
    "boolean": "bool",
    "integer": "int",
    "array": "List",
    "object": "Dict",
}


def is_cohere_model(llm: BaseLLM) -> bool:
    from llama_index.llms.cohere import Cohere

    return isinstance(llm, Cohere)


def is_cohere_function_calling_model(model_name: str) -> bool:
    return model_name in FUNCTION_CALLING_MODELS


logger = logging.getLogger(__name__)


# Cohere models accept a special argument `documents` for RAG calls. To pass on retrieved documents to the `documents` argument
# as intended by the Cohere team, we define:
# 1. A new ChatMessage class, called DocumentMessage
# 2. Specialised prompt templates for Text QA, Refine, Tree Summarize, and Refine Table that leverage DocumentMessage
# These templates are applied by default when a retriever is called with a Cohere LLM via custom logic inside default_prompt_selectors.py.
# See Cohere.chat for details on how the templates are unpackaged.


class DocumentMessage(ChatMessage):
    role: MessageRole = MessageRole.SYSTEM


# Define new templates with DocumentMessage's to leverage Cohere's `documents` argument
# Text QA
COHERE_QA_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        TEXT_QA_SYSTEM_PROMPT,
        DocumentMessage(content="{context_str}"),
        ChatMessage(content="{query_str}", role=MessageRole.USER),
    ]
)
# Refine (based on llama_index.core.chat_prompts::CHAT_REFINE_PROMPT)
REFINE_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that strictly operates in two modes "
        "when refining existing answers:\n"
        "1. **Rewrite** an original answer using the new context.\n"
        "2. **Repeat** the original answer if the new context isn't useful.\n"
        "Never reference the original answer or context directly in your answer.\n"
        "When in doubt, just repeat the original answer.\n"
    ),
    role=MessageRole.USER,
)
COHERE_REFINE_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        REFINE_SYSTEM_PROMPT,
        DocumentMessage(content="{context_msg}"),
        ChatMessage(
            content=(
                "Query: {query_str}\n"
                "Original Answer: {existing_answer}\n"
                "New Answer: "
            ),
            role=MessageRole.USER,
        ),
    ]
)
# Tree summarize (based on llama_index.core.chat_prompts::CHAT_TREE_SUMMARIZE_PROMPT)
COHERE_TREE_SUMMARIZE_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        TEXT_QA_SYSTEM_PROMPT,
        DocumentMessage(content="{context_str}"),
        ChatMessage(
            content=(
                "Given the information from multiple sources and not prior knowledge, "
                "answer the query.\n"
                "Query: {query_str}\n"
                "Answer: "
            ),
            role=MessageRole.USER,
        ),
    ]
)
# Table context refine (based on llama_index.core.chat_prompts::CHAT_REFINE_TABLE_CONTEXT_PROMPT)
COHERE_REFINE_TABLE_CONTEXT_PROMPT = ChatPromptTemplate(
    message_templates=[
        ChatMessage(content="{query_str}", role=MessageRole.USER),
        ChatMessage(content="{existing_answer}", role=MessageRole.ASSISTANT),
        DocumentMessage(content="{context_msg}"),
        ChatMessage(
            content=(
                "We have provided a table schema below. "
                "---------------------\n"
                "{schema}\n"
                "---------------------\n"
                "We have also provided some context information. "
                "Given the context information and the table schema, "
                "refine the original answer to better "
                "answer the question. "
                "If the context isn't useful, return the original answer."
            ),
            role=MessageRole.USER,
        ),
    ]
)


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    try:
        import cohere
    except ImportError as e:
        raise ImportError(
            "You must install the `cohere` package to use Cohere."
            "Please `pip install cohere`"
        ) from e

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(cohere.errors.ServiceUnavailableError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(
    client: Any, max_retries: int, chat: bool = False, **kwargs: Any
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        is_stream = kwargs.pop("stream", False)
        if chat:
            if is_stream:
                return client.chat_stream(**kwargs)
            else:
                return client.chat(**kwargs)
        else:
            if is_stream:
                return client.generate_stream(**kwargs)
            else:
                return client.generate(**kwargs)

    return _completion_with_retry(**kwargs)


async def acompletion_with_retry(
    aclient: Any,
    max_retries: int,
    chat: bool = False,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        is_stream = kwargs.pop("stream", False)
        if chat:
            if is_stream:
                return await aclient.chat_stream(**kwargs)
            else:
                return await aclient.chat(**kwargs)
        else:
            if is_stream:
                return await aclient.generate_stream(**kwargs)
            else:
                return await aclient.generate(**kwargs)

    return await _completion_with_retry(**kwargs)


def cohere_modelname_to_contextsize(modelname: str) -> int:
    context_size = ALL_AVAILABLE_MODELS.get(modelname, None)
    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Cohere model name."
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size


def is_chat_model(model: str) -> bool:
    return model in COMMAND_MODELS


def remove_documents_from_messages(
    messages: Sequence[ChatMessage],
) -> Tuple[Sequence[ChatMessage], Optional[List[Dict[str, str]]]]:
    """
    Splits messages into two lists: `remaining` and `documents`.
    `remaining` contains all messages that aren't of type DocumentMessage (e.g. history, query).
    `documents` contains the retrieved documents, formatted as expected by Cohere RAG inference calls.

    NOTE: this will mix turns for multi-turn RAG
    """
    documents = []
    remaining = []
    for msg in messages:
        if isinstance(msg, DocumentMessage):
            documents.append(msg)
        else:
            remaining.append(msg)
    return remaining, messages_to_cohere_documents(documents)


def messages_to_cohere_documents(
    messages: List[DocumentMessage],
) -> Optional[List[Dict[str, str]]]:
    """
    Splits out individual documents from `messages` in the format expected by Cohere.chat's `documents`.
    Returns None if `messages` is an empty list for compatibility with co.chat (where `documents` is None by default).
    """
    if messages == []:
        return None
    documents = []
    for msg in messages:
        documents.extend(document_message_to_cohere_document(msg))
    return documents


def _get_curr_chat_turn_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Get the messages for the current chat turn."""
    current_chat_turn_messages = []
    for message in messages[::-1]:
        current_chat_turn_messages.append(message)
        if message.role == MessageRole.USER:
            break
    return current_chat_turn_messages[::-1]


def _messages_to_cohere_tool_results_curr_chat_turn(
    messages: List[ChatMessage],
) -> List[Dict[str, Any]]:
    """Get tool_results from messages."""
    tool_results = []
    curr_chat_turn_messages = _get_curr_chat_turn_messages(messages)
    for message in curr_chat_turn_messages:
        if message.role == MessageRole.TOOL:
            tool_message = message
            previous_ai_msgs = [
                message
                for message in curr_chat_turn_messages
                if message.role in (MessageRole.CHATBOT, MessageRole.ASSISTANT)
                and message.additional_kwargs.get("tool_calls")
            ]
            if previous_ai_msgs:
                previous_ai_msg = previous_ai_msgs[-1]
                tool_results.extend(
                    [
                        {
                            "call": tool_call.__dict__,
                            "outputs": convert_to_documents(tool_message.content),
                        }
                        for tool_call in previous_ai_msg.additional_kwargs.get(
                            "tool_calls"
                        )
                    ]
                )

    return tool_results


def document_message_to_cohere_document(message: DocumentMessage) -> Dict:
    # By construction, single DocumentMessage contains all retrieved documents
    documents: List[Dict[str, str]] = []
    # Capture all key: value pairs. They will be unpacked in separate documents, Cohere-style.
    re_known_keywords = re.compile(
        r"(file_path|file_name|file_type|file_size|creation_date|last_modified_data|last_accessed_date): (.+)\n+"
    )
    # Find most frequent field. We assume that the most frequent field denotes the boundary
    # between consecutive documents, and break ties by taking whichever field appears first.
    known_keywords = re.findall(re_known_keywords, message.content)
    if len(known_keywords) == 0:
        # Document doesn't contain expected special fields. Return default formatting.
        return [{"text": message.content}]

    fields_counts = Counter([key for key, _ in known_keywords])
    most_frequent = fields_counts.most_common()[0][0]

    # Initialise
    document = None
    remaining_text = message.content
    for key, value in known_keywords:
        if key == most_frequent:
            # Save current document after extracting text, then reinit `document` to move to next document
            if document:  # skip first iteration, where document is None
                # Extract text up until the most_frequent remaining text, then skip to next document
                index = remaining_text.find(key)
                document["text"] = remaining_text[:index].strip()
                documents.append(document)
            document = {}

        # Catch all special fields. Convert them to key: value pairs.
        document[key] = value
        # Store remaining text, behind the (first instance of) current `value`
        remaining_text = remaining_text[
            remaining_text.find(value) + len(value) :
        ].strip()

    # Append last document that's in construction
    document["text"] = remaining_text.strip()
    documents.append(document)
    return documents


def _remove_signature_from_tool_description(name: str, description: str) -> str:
    """
    Removes the `{name}{signature} - ` prefix and Args: section from tool description.
    The Args: section may be present in function doc blocks.
    """
    description = re.sub(rf"^{name}\(.*?\) -(?:> \w+? -)? ", "", description)
    return re.sub(r"(?s)(?:\n?\n\s*?)?Args:.*$", "", description)


def format_to_cohere_tools(
    tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
) -> List[Dict[str, Any]]:
    return [_convert_to_cohere_tool(tool) for tool in tools]


def _convert_to_cohere_tool(
    tool: BaseTool,
) -> Dict[str, Any]:
    """
    Convert a BaseTool instance to a Cohere tool.
    """
    parameters = tool.metadata.get_parameters_dict()
    properties = parameters.get("properties", {})
    return Tool(
        name=tool.metadata.name,
        description=_remove_signature_from_tool_description(
            tool.metadata.name, tool.metadata.description
        ),
        parameter_definitions={
            param_name: ToolParameterDefinitionsValue(
                description=(
                    param_definition.get("description")
                    if "description" in param_definition
                    else ""
                ),
                type=JSON_TO_PYTHON_TYPES.get(
                    param_definition.get("type"), param_definition.get("type")
                ),
                required=param_name in parameters.get("required", []),
            )
            for param_name, param_definition in properties.items()
        },
    ).dict()


def convert_to_documents(
    observations: Any,
) -> List[MutableMapping]:
    """Converts observations into a 'document' dict."""
    documents: List[MutableMapping] = []
    if isinstance(observations, str):
        # strings are turned into a key/value pair and a key of 'output' is added.
        observations = [{"output": observations}]
    elif isinstance(observations, Mapping):
        # single mappings are transformed into a list to simplify the rest of the code.
        observations = [observations]
    elif not isinstance(observations, Sequence):
        # all other types are turned into a key/value pair within a list
        observations = [{"output": observations}]

    for doc in observations:
        if not isinstance(doc, Mapping):
            # types that aren't Mapping are turned into a key/value pair.
            doc = {"output": doc}
        documents.append(doc)

    return documents


def _message_to_cohere_tool_results(
    messages: List[ChatMessage], tool_message_index: int
) -> List[Dict[str, Any]]:
    """Get tool_results from messages."""
    tool_results = []
    tool_message = messages[tool_message_index]
    if not tool_message.role == MessageRole.TOOL:
        raise ValueError(
            "The message index does not correspond to an instance of ToolMessage"
        )

    messages_until_tool = messages[:tool_message_index]
    previous_ai_message = [
        message
        for message in messages_until_tool
        if message.role in (MessageRole.CHATBOT, MessageRole.ASSISTANT)
        and message.additional_kwargs.get("tool_calls")
    ][-1]
    tool_results.extend(
        [
            {
                "call": tool_call.__dict__,
                "outputs": convert_to_documents(tool_message.content),
            }
            for tool_call in previous_ai_message.additional_kwargs.get("tool_calls")
        ]
    )
    return tool_results


def get_role(message: ChatMessage) -> str:
    """Get the role of the message.

    Args:
        message: The message.

    Returns:
        The role of the message.

    Raises:
        ValueError: If the message is of an unknown type.
    """
    if message.role == MessageRole.USER:
        return "User"
    elif message.role == MessageRole.CHATBOT or message.role == MessageRole.ASSISTANT:
        return "Chatbot"
    elif message.role == MessageRole.SYSTEM:
        return "System"
    elif message.role == MessageRole.TOOL:
        return "Tool"
    else:
        raise ValueError(f"Got unknown type {type(message).__name__}")


def _get_message_cohere_format(
    message: ChatMessage, tool_results: Optional[List[Dict[Any, Any]]]
) -> Dict[
    str,
    Union[
        str,
        List[Union[str, Dict[Any, Any]]],
        List[Dict[Any, Any]],
        None,
    ],
]:
    """Get the formatted message as required in cohere's api.

    Args:
        message: The BaseMessage.
        tool_results: The tool results if any

    Returns:
        The formatted message as required in cohere's api.
    """
    if message.role == MessageRole.CHATBOT or message.role == MessageRole.ASSISTANT:
        return {
            "role": get_role(message),
            "message": message.content,
            "tool_calls": message.additional_kwargs.get("tool_calls"),
        }
    elif message.role in (MessageRole.USER, MessageRole.SYSTEM):
        return {"role": get_role(message), "message": message.content}
    elif message.role == MessageRole.TOOL:
        return {"role": get_role(message), "tool_results": tool_results}
    else:
        raise ValueError(f"Got unknown type {message}")
