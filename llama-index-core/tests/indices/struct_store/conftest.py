import re
from typing import Any, Dict, Optional, Tuple

import pytest
from tests.mock_utils.mock_prompts import (
    MOCK_REFINE_PROMPT,
    MOCK_SCHEMA_EXTRACT_PROMPT,
    MOCK_TEXT_QA_PROMPT,
)


def _mock_output_parser(output: str) -> Optional[Dict[str, Any]]:
    """
    Mock output parser.

    Split via commas instead of newlines, in order to fit
    the format of the mock test document (newlines create
    separate text chunks in the testing code).

    """
    tups = output.split(",")

    fields = {}
    for tup in tups:
        if ":" in tup:
            tokens = tup.split(":")
            field = re.sub(r"\W+", "", tokens[0])
            value = re.sub(r"\W+", "", tokens[1])
            fields[field] = value
    return fields


@pytest.fixture()
def struct_kwargs() -> Tuple[Dict, Dict]:
    """Index kwargs."""
    # NOTE: QuestionAnswer and Refine templates aren't technically used
    index_kwargs = {
        "schema_extract_prompt": MOCK_SCHEMA_EXTRACT_PROMPT,
        "output_parser": _mock_output_parser,
    }
    query_kwargs = {
        "text_qa_template": MOCK_TEXT_QA_PROMPT,
        "refine_template": MOCK_REFINE_PROMPT,
    }
    return index_kwargs, query_kwargs
