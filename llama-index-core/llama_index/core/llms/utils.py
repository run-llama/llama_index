from typing import TYPE_CHECKING, Optional, Union, Dict
import json
import re

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


_PARTIAL_JSON_STRUCTURAL_CHARS = re.compile(r'["{}\[\]]')
# Characters of a JSON string up to (not including) an unescaped quote, a raw
# newline or a trailing backslash.
_PARTIAL_JSON_STRING_BODY = re.compile(r'(?:[^"\\\n]+|\\[\s\S])*')


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
    pieces = []
    stack = []
    is_inside_string = False
    copied_up_to = 0
    pos = 0
    length = len(s)

    # Skip over string contents and non-structural characters in bulk instead
    # of walking the input one character at a time.
    while pos < length:
        if is_inside_string:
            pos = _PARTIAL_JSON_STRING_BODY.match(s, pos).end()
            if pos >= length:
                break
            char = s[pos]
            if char == '"':
                is_inside_string = False
            elif char == "\n":
                # Replace the newline character with the escape sequence.
                pieces.append(s[copied_up_to:pos])
                pieces.append("\\n")
                copied_up_to = pos + 1
            # Otherwise it is a trailing backslash at the end of the input.
            pos += 1
        else:
            match = _PARTIAL_JSON_STRUCTURAL_CHARS.search(s, pos)
            if match is None:
                break
            char = match.group()
            pos = match.end()
            if char == '"':
                is_inside_string = True
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif stack and stack[-1] == char:
                stack.pop()
            else:
                # Mismatched closing character; the input is malformed.
                raise ValueError("Malformed partial JSON encountered.")

    pieces.append(s[copied_up_to:])
    new_s = "".join(pieces)

    # If we're still inside a string at the end of processing and no colon was found after the opening quote,
    # this is an incomplete key - remove it
    if is_inside_string and '"' in new_s and ":" not in new_s[new_s.rindex('"') :]:
        new_s = new_s[: new_s.rindex('"')]
    elif is_inside_string:
        new_s += '"'

    # Check if we have an incomplete key-value pair
    new_s = new_s.rstrip()
    if new_s.endswith(":"):
        new_s += " null"  # Add a default value for incomplete value
    elif new_s.endswith(","):
        new_s = new_s[:-1]  # Remove the trailing comma

    # Close any remaining open structures in the reverse order that they were opened.
    for closing_char in reversed(stack):
        new_s += closing_char

    # Attempt to parse the modified string as JSON.
    try:
        return json.loads(new_s)
    except json.JSONDecodeError:
        # If we still can't parse the string as JSON, raise error to indicate failure.
        raise ValueError("Malformed partial JSON encountered.")
