from typing import TYPE_CHECKING, List, Optional, Union, Dict
import json

if TYPE_CHECKING:
    try:
        # For langchain v1.x.x
        from langchain_core.language_models import (
            BaseLanguageModel,
        )  # pants: no-infer-dep
    except ImportError:
        # For langchain v0.x.x
        from langchain.base_language import BaseLanguageModel  # pants: no-infer-dep

import os

from llama_index.core.llms.callbacks import CallbackManager
from llama_index.core.llms.llm import LLM

LLMType = Union[str, LLM, "BaseLanguageModel"]


def resolve_llm(
    llm: Optional[LLMType] = None, callback_manager: Optional[CallbackManager] = None
) -> LLM:
    """Resolve LLM from string or LLM instance."""
    from llama_index.core.settings import Settings

    try:
        # For langchain v1.x.x
        from langchain_core.language_models import (
            BaseLanguageModel,
        )  # pants: no-infer-dep
    except ImportError:
        try:
            # For langchain v0.x.x
            from langchain.base_language import BaseLanguageModel  # pants: no-infer-dep
        except ImportError:
            BaseLanguageModel = None  # type: ignore

    if llm == "default":
        # if testing return mock llm
        if os.getenv("IS_TESTING"):
            from llama_index.core.llms.mock import MockLLM

            llm = MockLLM()
            llm.callback_manager = callback_manager or Settings.callback_manager
            return llm

        # return default OpenAI model. If it fails, return LlamaCPP
        try:
            from llama_index.llms.openai import OpenAI  # pants: no-infer-dep
            from llama_index.llms.openai.utils import (
                validate_openai_api_key,
            )  # pants: no-infer-dep

            llm = OpenAI()
            validate_openai_api_key(llm.api_key)  # type: ignore
        except ImportError:
            raise ImportError(
                "`llama-index-llms-openai` package not found, "
                "please run `pip install llama-index-llms-openai`"
            )
        except ValueError as e:
            raise ValueError(
                "\n******\n"
                "Could not load OpenAI model. "
                "If you intended to use OpenAI, please check your OPENAI_API_KEY.\n"
                "Original error:\n"
                f"{e!s}"
                "\n******"
            )

    if isinstance(llm, str):
        splits = llm.split(":", 1)
        is_local = splits[0]
        model_path = splits[1] if len(splits) > 1 else None
        if is_local != "local":
            raise ValueError(
                "llm must start with str 'local' or of type LLM or BaseLanguageModel"
            )
        try:
            from llama_index.llms.llama_cpp.llama_utils import (
                completion_to_prompt,
                messages_to_prompt,
            )  # pants: no-infer-dep

            from llama_index.llms.llama_cpp import LlamaCPP  # pants: no-infer-dep

            llm = LlamaCPP(
                model_path=model_path,
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                model_kwargs={"n_gpu_layers": 1},
            )
        except ImportError:
            raise ImportError(
                "`llama-index-llms-llama-cpp` package not found, "
                "please run `pip install llama-index-llms-llama-cpp`"
            )

    elif BaseLanguageModel is not None and isinstance(llm, BaseLanguageModel):
        # NOTE: if it's a langchain model, wrap it in a LangChainLLM
        try:
            from llama_index.llms.langchain import LangChainLLM  # pants: no-infer-dep

            llm = LangChainLLM(llm=llm)
        except ImportError:
            raise ImportError(
                "`llama-index-llms-langchain` package not found, "
                "please run `pip install llama-index-llms-langchain`"
            )
    elif llm is None:
        from llama_index.core.llms.mock import MockLLM

        print("LLM is explicitly disabled. Using MockLLM.")
        llm = MockLLM()

    assert isinstance(llm, LLM)

    llm.callback_manager = (
        callback_manager or llm.callback_manager or Settings.callback_manager
    )

    return llm


def _terminate_partial_string(s: str, string_start: int) -> str:
    r"""
    Close the unterminated string token that starts at ``string_start``.

    The tail of the token may be a truncated escape sequence (a lone ``\`` or a
    cut ``\uXXXX``), which cannot be closed as-is. Trim from the end until the
    token parses, which is at most a handful of characters.
    """
    token = s[string_start:]
    while not token.endswith('"'):
        try:
            json.loads(token + '"')
            break
        except json.JSONDecodeError:
            token = token[:-1]

    return s[:string_start] + token + '"'


def parse_partial_json(s: str) -> Dict:
    """
    Parse an incomplete JSON string into a valid python dictionary.

    NOTE: This is adapted from
    https://github.com/OpenInterpreter/open-interpreter/blob/5b6080fae1f8c68938a1e4fa8667e3744084ee21/interpreter/utils/parse_partial_json.py
    """
    # Attempt to parse the string as-is.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Initialize variables.
    new_s = ""
    stack: List[str] = []
    is_inside_string = False
    escaped = False
    # Where the string token being scanned starts, and whether it is an object key.
    string_start = -1
    string_is_key = False
    # Start of an object key that has no ":" after it yet.
    dangling_key_start = -1
    # Start of a bare literal (number/true/false/null) that may be cut mid-token.
    literal_start = -1
    # Last structural character seen outside a string, used to tell a key from a value.
    last_structural = ""

    # Process each character in the string one at a time.
    for char in s:
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
                if string_is_key:
                    dangling_key_start = string_start
            elif char == "\n" and not escaped:
                char = "\\n"  # Replace the newline character with the escape sequence.
            elif char == "\\":
                escaped = not escaped
            else:
                escaped = False
        else:
            if char == '"':
                is_inside_string = True
                escaped = False
                string_start = len(new_s)
                # Only a string sitting in an object right after "{" or "," is a key.
                string_is_key = (
                    bool(stack) and stack[-1] == "}" and last_structural in ("{", ",")
                )
                literal_start = -1
            elif char == "{":
                stack.append("}")
                last_structural = char
                literal_start = -1
            elif char == "[":
                stack.append("]")
                last_structural = char
                literal_start = -1
            elif char == "}" or char == "]":
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    # Mismatched closing character; the input is malformed.
                    raise ValueError("Malformed partial JSON encountered.")
                last_structural = ""
                literal_start = -1
            elif char == ":":
                last_structural = char
                dangling_key_start = -1
                literal_start = -1
            elif char == ",":
                last_structural = char
                literal_start = -1
            elif not char.isspace() and literal_start == -1:
                literal_start = len(new_s)

        # Append the processed character to the new string.
        new_s += char

    if is_inside_string:
        if string_is_key:
            # An incomplete key carries no information at all -- drop it.
            dangling_key_start = string_start
        else:
            # An incomplete value does, so keep what has arrived so far.
            new_s = _terminate_partial_string(new_s, string_start)

    if dangling_key_start != -1:
        # A key with no value yet; drop it along with the comma introducing it.
        new_s = new_s[:dangling_key_start]
        literal_start = -1

    def close(text: str) -> str:
        # Check if we have an incomplete key-value pair
        text = text.rstrip()
        if text.endswith(":"):
            text += " null"  # Add a default value for incomplete value
        elif text.endswith(","):
            text = text[:-1]  # Remove the trailing comma

        # Close any remaining open structures in the reverse order they were opened.
        return text + "".join(reversed(stack))

    attempts = [new_s]
    if literal_start != -1:
        # A trailing literal may be a value cut mid-token ("1.", "tru", "-"),
        # which is only detectable by failing to parse it.
        attempts.append(new_s[:literal_start])

    for attempt in attempts:
        # Attempt to parse the modified string as JSON.
        try:
            return json.loads(close(attempt))
        except json.JSONDecodeError:
            continue

    # If we still can't parse the string as JSON, raise error to indicate failure.
    raise ValueError("Malformed partial JSON encountered.")
